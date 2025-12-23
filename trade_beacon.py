#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI FOREX BRAIN - COMPLETE ELITE TRADING SYSTEM WITH STRICT API LIMITS
======================================================================
✅ Single-file, production-ready
✅ Environment-aware (Colab/GHA/Local)
✅ Dashboard-controlled
✅ Multi-source price rotation
✅ News & economic calendar aware
✅ Learning system with memory
✅ FIXED: Historical data fetching with proper symbol format
✅ FIXED: EconomicCalendar.get_upcoming_events() method signature
✅ NEW: Strict API rate limiting for all services
✅ NEW: Browserless resumes Jan 19th with 5 calls/day limit
✅ NEW: Alpha Vantage 25 calls/day limit (Free tier: 500/day but conservative)
✅ NEW: YFinance 100 calls/day limit (prevent abuse)
✅ NEW: Marketaux 90 calls/day limit (Free tier: 100/day)
"""

# ======================================================
# 🔐 SECTION 1: API Keys Configuration
# ======================================================
import os
import sys
from pathlib import Path

ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '1W58NPZXOG5SLHZ6')
BROWSERLESS_TOKEN = os.environ.get('BROWSERLESS_TOKEN', '2TMVUBAjFwrr7Tb283f0da6602a4cb698b81778bda61967f7')

os.environ['ALPHA_VANTAGE_KEY'] = ALPHA_VANTAGE_KEY
os.environ['BROWSERLESS_TOKEN'] = BROWSERLESS_TOKEN

# ======================================================
# 🌍 SECTION 2: Environment Detection & Setup
# ======================================================
print("=" * 70)
print("🚀 AI FOREX BRAIN - INITIALIZING...")
print("=" * 70)

# Detect environment
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

# Set base paths
if IN_COLAB:
    BASE_FOLDER = Path("/content")
    SAVE_FOLDER = BASE_FOLDER / "forex-ai-models"
elif IN_GHA:
    BASE_FOLDER = Path.cwd()
    SAVE_FOLDER = BASE_FOLDER
else:
    BASE_FOLDER = Path.cwd()
    SAVE_FOLDER = BASE_FOLDER

# Create organized directory structure
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
print(f"🔧 Python: {sys.version.split()[0]}")
print("=" * 70)

# Export path constants
CSV_FOLDER = DIRECTORIES["data_raw"]
PICKLE_FOLDER = DIRECTORIES["data_processed"]
DB_PATH = DIRECTORIES["database"] / "memory_v85.db"
LOG_PATH = DIRECTORIES["logs"] / "pipeline.log"
OUTPUT_PATH = DIRECTORIES["outputs"] / "signals.json"
MEMORY_DIR = DIRECTORIES["memory"]
STATE_DIR = DIRECTORIES["signal_state"]

# ======================================================
# 📂 SECTION 3: GitHub Sync
# ======================================================
import subprocess
import shutil
import urllib.parse

GITHUB_USERNAME = "rahim-dotAI"
GITHUB_REPO = "forex-ai-models"
BRANCH = "main"
REPO_FOLDER = SAVE_FOLDER

# Get GitHub token
FOREX_PAT = os.environ.get("FOREX_PAT")

if not FOREX_PAT and IN_COLAB:
    try:
        from google.colab import userdata
        FOREX_PAT = userdata.get("FOREX_PAT")
        if FOREX_PAT:
            os.environ["FOREX_PAT"] = FOREX_PAT
            print("🔐 Loaded FOREX_PAT from Colab secret.")
    except:
        pass

REPO_URL = None
if FOREX_PAT:
    SAFE_PAT = urllib.parse.quote(FOREX_PAT)
    REPO_URL = f"https://{GITHUB_USERNAME}:{SAFE_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
    print("✅ GitHub token configured")
else:
    print("⚠️ Warning: FOREX_PAT not found. Git operations may fail.")

# Handle repository based on environment
if IN_GHA:
    print("\n🤖 GitHub Actions Mode")
    print("✅ Repository already checked out by actions/checkout")
    
elif IN_COLAB:
    print("\n☁️ Google Colab Mode")
    
    if not REPO_URL:
        print("❌ Cannot clone repository: FOREX_PAT not available")
    elif not (REPO_FOLDER / ".git").exists():
        if REPO_FOLDER.exists() and not (REPO_FOLDER / ".git").exists():
            print(f"⚠️ Directory exists but is not a git repo. Removing...")
            shutil.rmtree(REPO_FOLDER)
        
        print(f"📥 Cloning repository to {REPO_FOLDER}...")
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        
        try:
            subprocess.run(
                ["git", "clone", "-b", BRANCH, REPO_URL, str(REPO_FOLDER)],
                check=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            print("✅ Repository cloned successfully")
            os.chdir(REPO_FOLDER)
            print(f"📂 Changed directory to: {os.getcwd()}")
        except Exception as e:
            print(f"⚠️ Clone failed: {e}")
            REPO_FOLDER.mkdir(parents=True, exist_ok=True)
    else:
        print("✅ Repository already exists, pulling latest changes...")
        os.chdir(REPO_FOLDER)
        try:
            subprocess.run(
                ["git", "pull", "origin", BRANCH],
                check=True,
                cwd=REPO_FOLDER,
                capture_output=True,
                text=True,
                timeout=30
            )
            print("✅ Successfully pulled latest changes")
        except Exception as e:
            print(f"⚠️ Pull failed: {e}")
    
    # Disable Git LFS for Colab
    try:
        subprocess.run(
            ["git", "lfs", "uninstall"],
            check=False,
            cwd=REPO_FOLDER,
            capture_output=True
        )
        print("✅ LFS disabled for Colab")
    except:
        pass
else:
    print("\n💻 Local Development Mode")
    print(f"📂 Working in: {SAVE_FOLDER}")

# Configure Git
GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")

git_configs = [
    (["git", "config", "--global", "user.name", GIT_USER_NAME], "User name"),
    (["git", "config", "--global", "user.email", GIT_USER_EMAIL], "User email"),
    (["git", "config", "--global", "advice.detachedHead", "false"], "Detached HEAD warning"),
    (["git", "config", "--global", "init.defaultBranch", "main"], "Default branch")
]

for cmd, description in git_configs:
    try:
        subprocess.run(cmd, check=False, capture_output=True)
    except:
        pass

print(f"✅ Git configured: {GIT_USER_NAME} <{GIT_USER_EMAIL}>")

print("\n" + "=" * 70)
print("✅ ENVIRONMENT SETUP COMPLETED")
print("=" * 70)
print(f"✅ API Keys: {ALPHA_VANTAGE_KEY[:4]}...{ALPHA_VANTAGE_KEY[-4:]}")
print(f"✅ Directories: {len(DIRECTORIES)} created")
print(f"✅ Git Status: {'Configured' if (REPO_FOLDER / '.git').exists() else 'Not a repo'}")
print("=" * 70)

# ======================================================
# 📦 SECTION 4: Install Dependencies
# ======================================================
print("\n📦 Installing dependencies...")
if IN_COLAB or IN_GHA:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests", "pandas", "numpy", "yfinance", "tqdm"], check=False)
    print("✅ Dependencies installed")
else:
    print("💻 Local mode: Ensure dependencies are installed")
    print("   pip install requests pandas numpy yfinance tqdm")

# ======================================================
# 🎯 SECTION 5: TRADE BEACON - Complete Trading System
# ======================================================
print("\n" + "=" * 70)
print("🎯 LOADING TRADE BEACON SYSTEM WITH STRICT API LIMITS...")
print("=" * 70)

import json
import time
import requests
import threading
import yfinance as yf
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import deque
import logging
import traceback

# State files
SYSTEM_STATE_FILE = STATE_DIR / "system_state.json"
ACTIVE_SIGNALS_FILE = STATE_DIR / "active_signals.json"
DASHBOARD_STATE_FILE = STATE_DIR / "dashboard_state.json"
LEARNING_MEMORY_FILE = MEMORY_DIR / "learning_memory.json"
TRADE_HISTORY_FILE = MEMORY_DIR / "trade_history.json"
PRICE_SOURCE_STATE_FILE = STATE_DIR / "price_source_rotation.json"
API_RATE_LIMIT_FILE = STATE_DIR / "api_rate_limits.json"

# Currency pairs
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"]

# API Keys - Load from environment variables
MARKETAUX_API_KEY = os.environ.get('MARKETAUX_API_KEY', '')

# Signal parameters
MAX_ACTIVE_SIGNALS = 5
ATR_SL_MULT = 2.0
ATR_TP_MULT = 3.0
MIN_CONFIDENCE = 0.72
SIGNAL_CHECK_INTERVAL = 15

# ============================================================================
# STRICT API RATE LIMITING CONFIGURATION
# ============================================================================
API_LIMITS = {
    'yfinance': {
        'daily_limit': 100,
        'description': 'YFinance (prevent abuse)',
        'enabled': True
    },
    'alpha_vantage': {
        'daily_limit': 25,
        'description': 'Alpha Vantage (Free tier: 500/day, using 25 for safety)',
        'enabled': True
    },
    'browserless': {
        'daily_limit': 5,
        'description': 'Browserless (5 calls/day after Jan 19)',
        'enabled': False,  # Will be enabled on Jan 19, 2025
        'enable_date': datetime(2025, 1, 19, 0, 0, 0, tzinfo=timezone.utc)
    },
    'marketaux': {
        'daily_limit': 90,
        'description': 'Marketaux (Free tier: 100/day, using 90 for safety)',
        'enabled': True
    }
}

NEWS_CHECK_INTERVAL = 900  # 15 minutes
CALENDAR_CHECK_INTERVAL = 3600  # 1 hour
PRICE_CACHE_DURATION = 60

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not MARKETAUX_API_KEY:
    logger.warning("⚠️ MARKETAUX_API_KEY not set - news features will be limited")

# ============================================================================
# CENTRALIZED API RATE LIMITER
# ============================================================================
class APIRateLimiter:
    """Centralized API rate limiting for all services"""
    
    def __init__(self):
        self.limits = API_LIMITS.copy()
        self.calls = {}
        self.last_reset_date = None
        self.load_state()
        self.check_browserless_enable()
        
    def load_state(self):
        """Load API call counters from file"""
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
                        # New day, reset counters
                        self.reset_counters()
            except:
                self.reset_counters()
        else:
            self.reset_counters()
    
    def save_state(self):
        """Save API call counters to file"""
        with open(API_RATE_LIMIT_FILE, 'w') as f:
            json.dump({
                'date': datetime.now(timezone.utc).date().isoformat(),
                'calls': self.calls,
                'limits': self.limits,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
    
    def reset_counters(self):
        """Reset daily counters"""
        self.calls = {api: 0 for api in self.limits.keys()}
        self.last_reset_date = datetime.now(timezone.utc).date().isoformat()
        self.save_state()
        logger.info("🔄 API rate limit counters reset for new day")
    
    def check_browserless_enable(self):
        """Check if Browserless should be enabled on Jan 19, 2025"""
        now = datetime.now(timezone.utc)
        enable_date = self.limits['browserless']['enable_date']
        
        if now >= enable_date and not self.limits['browserless']['enabled']:
            self.limits['browserless']['enabled'] = True
            logger.info("✅ BROWSERLESS API ENABLED (Jan 19, 2025 reached)")
            self.save_state()
        elif now < enable_date:
            logger.info(f"⏰ Browserless will be enabled on: {enable_date.strftime('%Y-%m-%d %H:%M UTC')}")
    
    def can_make_call(self, api_name: str) -> Tuple[bool, str]:
        """Check if API call is allowed within limits"""
        today = datetime.now(timezone.utc).date().isoformat()
        
        # Reset if new day
        if self.last_reset_date != today:
            self.reset_counters()
        
        # Check if API exists
        if api_name not in self.limits:
            return False, f"Unknown API: {api_name}"
        
        # Check if API is enabled
        if not self.limits[api_name]['enabled']:
            enable_date = self.limits[api_name].get('enable_date')
            if enable_date:
                return False, f"{api_name} disabled until {enable_date.strftime('%Y-%m-%d')}"
            return False, f"{api_name} is disabled"
        
        # Check daily limit
        current_calls = self.calls.get(api_name, 0)
        limit = self.limits[api_name]['daily_limit']
        
        if current_calls >= limit:
            return False, f"{api_name} daily limit reached ({current_calls}/{limit})"
        
        return True, f"OK ({current_calls}/{limit})"
    
    def record_call(self, api_name: str, success: bool = True):
        """Record an API call"""
        if api_name not in self.calls:
            self.calls[api_name] = 0
        
        self.calls[api_name] += 1
        self.save_state()
        
        current = self.calls[api_name]
        limit = self.limits[api_name]['daily_limit']
        percentage = (current / limit) * 100
        
        if success:
            logger.debug(f"✅ {api_name}: {current}/{limit} ({percentage:.0f}%)")
        else:
            logger.debug(f"❌ {api_name} call failed: {current}/{limit} ({percentage:.0f}%)")
        
        # Warning when approaching limit
        if percentage >= 80:
            logger.warning(f"⚠️ {api_name} at {percentage:.0f}% of daily limit!")
    
    def get_stats(self) -> Dict:
        """Get comprehensive API usage statistics"""
        stats = {}
        for api_name, config in self.limits.items():
            current = self.calls.get(api_name, 0)
            limit = config['daily_limit']
            stats[api_name] = {
                'enabled': config['enabled'],
                'calls': current,
                'limit': limit,
                'remaining': limit - current,
                'percentage': (current / limit * 100) if limit > 0 else 0,
                'description': config['description']
            }
            if 'enable_date' in config and not config['enabled']:
                stats[api_name]['enable_date'] = config['enable_date'].isoformat()
        
        return stats
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = ["📊 API Usage Summary:"]
        for api_name, stats in self.get_stats().items():
            status = "✅" if stats['enabled'] else "❌"
            lines.append(
                f"{status} {api_name}: {stats['calls']}/{stats['limit']} "
                f"({stats['percentage']:.0f}%) - {stats['description']}"
            )
            if 'enable_date' in stats:
                lines.append(f"   ⏰ Enables: {stats['enable_date']}")
        return "\n".join(lines)

# Initialize global rate limiter
api_limiter = APIRateLimiter()

# ============================================================================
# PRICE SOURCE ROTATION MANAGER (WITH STRICT LIMITS)
# ============================================================================
class PriceSourceRotation:
    """Intelligent price source rotation with strict API limits"""

    def __init__(self):
        self.sources = {
            'yfinance': {'weight': 10, 'calls': 0, 'failures': 0, 'blocked': 0},
            'alpha_vantage': {'weight': 20, 'calls': 0, 'failures': 0, 'blocked': 0},
            'browserless': {'weight': 5, 'calls': 0, 'failures': 0, 'blocked': 0}
        }
        self.total_weight = 35
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
        with open(PRICE_SOURCE_STATE_FILE, 'w') as f:
            json.dump({
                'sources': self.sources,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    def get_next_source(self) -> Optional[str]:
        """Get next available source respecting API limits"""
        available_sources = []
        
        for source in ['yfinance', 'alpha_vantage', 'browserless']:
            can_call, reason = api_limiter.can_make_call(source)
            if can_call:
                available_sources.append(source)
            else:
                logger.debug(f"⚠️ {source} unavailable: {reason}")
        
        if not available_sources:
            logger.warning("❌ All API sources exhausted for today!")
            return None
        
        # Weighted selection from available sources
        total_calls = sum(self.sources[s]['calls'] for s in available_sources)
        total_weight = sum(self.sources[s]['weight'] for s in available_sources)
        
        for source in available_sources:
            expected_calls = (self.sources[source]['weight'] / total_weight) * total_calls if total_calls > 0 else 0
            if self.sources[source]['calls'] < expected_calls or total_calls == 0:
                return source
        
        return available_sources[0]

    def record_call(self, source: str, success: bool, blocked: bool = False):
        """Record call with blocking status"""
        if source in self.sources:
            self.sources[source]['calls'] += 1
            if not success:
                self.sources[source]['failures'] += 1
            if blocked:
                self.sources[source]['blocked'] += 1
            self.save_state()

    def get_cached_price(self, pair: str) -> Optional[float]:
        if pair in self.cache:
            age = time.time() - self.cache[pair]['timestamp']
            if age < PRICE_CACHE_DURATION:
                return self.cache[pair]['price']
        return None

    def cache_price(self, pair: str, price: float, source: str):
        self.cache[pair] = {
            'price': price,
            'timestamp': time.time(),
            'source': source
        }

    def get_stats(self) -> Dict:
        total_calls = sum(s['calls'] for s in self.sources.values())
        return {
            'total_calls': total_calls,
            'by_source': {
                source: {
                    'calls': stats['calls'],
                    'percentage': (stats['calls'] / total_calls * 100) if total_calls > 0 else 0,
                    'failures': stats['failures'],
                    'blocked': stats['blocked'],
                    'success_rate': ((stats['calls'] - stats['failures']) / stats['calls'] * 100)
                                   if stats['calls'] > 0 else 0
                }
                for source, stats in self.sources.items()
            }
        }

price_rotation = PriceSourceRotation()

# ============================================================================
# MULTI-SOURCE PRICE FETCHER (WITH STRICT LIMITS)
# ============================================================================
def fetch_price_yfinance(pair: str) -> Optional[float]:
    """Fetch live price from YFinance with rate limiting"""
    can_call, reason = api_limiter.can_make_call('yfinance')
    if not can_call:
        logger.debug(f"YFinance blocked: {reason}")
        price_rotation.record_call('yfinance', False, blocked=True)
        return None
    
    try:
        symbol = pair.replace('/', '') + '=X'
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')

        if not data.empty:
            price = float(data['Close'].iloc[-1])
            api_limiter.record_call('yfinance', True)
            logger.debug(f"📊 YFinance: {pair} = {price:.5f}")
            return price
        else:
            api_limiter.record_call('yfinance', False)
    except Exception as e:
        api_limiter.record_call('yfinance', False)
        logger.debug(f"YFinance error for {pair}: {e}")
    return None

def fetch_price_alpha_vantage(pair: str) -> Optional[float]:
    """Fetch live price from Alpha Vantage with rate limiting"""
    if not ALPHA_VANTAGE_KEY:
        return None
    
    can_call, reason = api_limiter.can_make_call('alpha_vantage')
    if not can_call:
        logger.debug(f"Alpha Vantage blocked: {reason}")
        price_rotation.record_call('alpha_vantage', False, blocked=True)
        return None

    try:
        from_curr, to_curr = pair.split('/')
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_curr,
            'to_currency': to_curr,
            'apikey': ALPHA_VANTAGE_KEY
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'Realtime Currency Exchange Rate' in data:
            price = float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
            api_limiter.record_call('alpha_vantage', True)
            logger.debug(f"🔑 Alpha Vantage: {pair} = {price:.5f}")
            return price
        else:
            api_limiter.record_call('alpha_vantage', False)
    except Exception as e:
        api_limiter.record_call('alpha_vantage', False)
        logger.debug(f"Alpha Vantage error for {pair}: {e}")
    return None

def fetch_price_browserless(pair: str) -> Optional[float]:
    """Fetch live price from X-Rates via Browserless with rate limiting"""
    if not BROWSERLESS_TOKEN:
        return None
    
    can_call, reason = api_limiter.can_make_call('browserless')
    if not can_call:
        logger.debug(f"Browserless blocked: {reason}")
        price_rotation.record_call('browserless', False, blocked=True)
        return None

    try:
        from_curr, to_curr = pair.split("/")
        url = f"https://www.x-rates.com/calculator/?from={from_curr}&to={to_curr}&amount=1"

        response = requests.post(
            f"https://production-sfo.browserless.io/content?token={BROWSERLESS_TOKEN}",
            json={"url": url},
            timeout=15
        )

        if response.status_code == 200:
            import re
            match = re.search(r'ccOutputRslt[^>]*>([\d,.]+)', response.text)
            if match:
                price = float(match.group(1).replace(",", ""))
                api_limiter.record_call('browserless', True)
                logger.debug(f"🌐 Browserless: {pair} = {price:.5f}")
                return price
            else:
                api_limiter.record_call('browserless', False)
        else:
            api_limiter.record_call('browserless', False)
    except Exception as e:
        api_limiter.record_call('browserless', False)
        logger.debug(f"Browserless error for {pair}: {e}")
    return None

def fetch_live_price(pair: str) -> float:
    """Fetch live price with intelligent source rotation and strict limits"""
    cached_price = price_rotation.get_cached_price(pair)
    if cached_price:
        return cached_price

    source = price_rotation.get_next_source()
    
    if not source:
        logger.warning(f"⚠️ All API sources exhausted, using historical data for {pair}")
        return get_historical_price(pair)

    price = None
    if source == 'yfinance':
        price = fetch_price_yfinance(pair)
    elif source == 'alpha_vantage':
        price = fetch_price_alpha_vantage(pair)
    elif source == 'browserless':
        price = fetch_price_browserless(pair)

    if price:
        price_rotation.record_call(source, True)
        price_rotation.cache_price(pair, price, source)
        return price
    else:
        price_rotation.record_call(source, False)

    fallback_sources = [s for s in ['yfinance', 'alpha_vantage', 'browserless'] if s != source]

    for fallback in fallback_sources:
        can_call, _ = api_limiter.can_make_call(fallback)
        if not can_call:
            continue
            
        if fallback == 'yfinance':
            price = fetch_price_yfinance(pair)
        elif fallback == 'alpha_vantage':
            price = fetch_price_alpha_vantage(pair)
        elif fallback == 'browserless':
            price = fetch_price_browserless(pair)

        if price:
            price_rotation.record_call(fallback, True)
            price_rotation.cache_price(pair, price, fallback)
            return price
        else:
            price_rotation.record_call(fallback, False)

    logger.warning(f"⚠️ All sources failed for {pair}, using historical data")
    return get_historical_price(pair)

def get_historical_price(pair: str) -> float:
    """Get latest price from historical data files"""
    try:
        pkl_files = list(PICKLE_FOLDER.glob(f"*{pair.replace('/', '_')}*.pkl"))
        if pkl_files:
            df = pd.read_pickle(pkl_files[0])
            if len(df) > 0:
                return float(df['close'].iloc[-1])
    except Exception as e:
        logger.error(f"Error loading historical data for {pair}: {e}")

    defaults = {
        'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 149.50,
        'AUD/USD': 0.6550, 'USD/CAD': 1.3450, 'NZD/USD': 0.6150
    }
    return defaults.get(pair, 1.0)

# ============================================================================
# HISTORICAL DATA FETCHER (FIXED)
# ============================================================================
def fetch_historical_data(pair: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch or load historical forex data with proper symbol formatting"""
    pair_name = pair.replace('/', '_')
    cache_file = PICKLE_FOLDER / f"{pair_name}_{interval}_{period}.pkl"

    # Check cache
    if cache_file.exists():
        try:
            df = pd.read_pickle(cache_file)
            if len(df) > 0 and (datetime.now() - df.index[-1]).days < 1:
                logger.debug(f"💾 Loaded cached data for {pair}")
                return df
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")

    # Fetch fresh data
    try:
        symbol = pair.replace('/', '') + '=X'
        logger.info(f"📥 Fetching {period} data for {pair} (symbol: {symbol})...")
        
        df = yf.download(
            symbol, 
            period=period, 
            interval=interval, 
            progress=False,
            auto_adjust=True
        )

        if df is None or df.empty:
            logger.error(f"❌ No data returned for {pair} (symbol: {symbol})")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing columns for {pair}: {missing_cols}")
            logger.error(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        df.to_pickle(cache_file)
        logger.info(f"✅ Fetched {len(df)} candles for {pair}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch data for {pair}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
class TechnicalIndicators:
    """Complete technical indicators library"""

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
    def stochastic(df: pd.DataFrame, k_period=14, d_period=3):
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(data: pd.Series, period=20, std_dev=2):
        middle = data.rolling(period).mean()
        std = data.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

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

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR value"""
    try:
        atr_series = TechnicalIndicators.atr(df, period)
        atr_value = atr_series.iloc[-1]
        return float(atr_value) if not np.isnan(atr_value) else 0.001
    except:
        return 0.001

# ============================================================================
# SYSTEM CONTROLLER
# ============================================================================
class SystemController:
    """Controls system start/stop from dashboard"""

    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.start_time = None
        self.load_state()

    def load_state(self):
        if SYSTEM_STATE_FILE.exists():
            try:
                with open(SYSTEM_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.is_running = data.get('is_running', False)
                    self.start_time = data.get('start_time')
            except:
                pass

    def save_state(self):
        with open(SYSTEM_STATE_FILE, 'w') as f:
            json.dump({
                'is_running': self.is_running,
                'should_stop': self.should_stop,
                'start_time': self.start_time,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    def start(self):
        self.is_running = True
        self.should_stop = False
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.save_state()
        logger.info("🚀 SYSTEM STARTED by dashboard")

    def stop(self):
        self.should_stop = True
        self.is_running = False
        self.save_state()
        logger.info("⏸️ SYSTEM STOPPED by dashboard")

    def check_should_run(self) -> Tuple[bool, str]:
        now = datetime.now(timezone.utc)

        if self.should_stop:
            return False, "Manual stop requested"

        if now.weekday() in [5, 6]:
            return False, "Weekend - Markets closed"

        holidays = [(1, 1), (12, 25), (12, 26)]
        if (now.month, now.day) in holidays:
            return False, "Holiday - Markets closed"

        if now.weekday() == 4 and now.hour >= 22:
            return False, "Week end - Markets closing"

        return True, "Running normally"

# ============================================================================
# NEWS & SENTIMENT ANALYZER (WITH STRICT API RATE LIMITING)
# ============================================================================
class NewsAnalyzer:
    """Fetches and analyzes financial news + sentiment with strict API rate limiting"""

    def __init__(self):
        self.last_news_check = 0
        self.news_cache = []
        self.high_impact_events = []

    def fetch_news(self) -> List[Dict]:
        """Fetch news with strict rate limiting"""
        if not MARKETAUX_API_KEY:
            return self.news_cache

        can_call, reason = api_limiter.can_make_call('marketaux')
        if not can_call:
            logger.debug(f"Marketaux blocked: {reason}")
            return self.news_cache

        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                'api_token': MARKETAUX_API_KEY,
                'symbols': 'EUR,USD,GBP,JPY,AUD,CAD,NZD',
                'filter_entities': 'true',
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
                return self.news_cache
            elif response.status_code in [402, 429]:
                api_limiter.record_call('marketaux', False)
                logger.warning(f"⚠️ Marketaux API error: {response.status_code}")
                self.last_news_check = time.time()
                return self.news_cache
            else:
                api_limiter.record_call('marketaux', False)
                self.last_news_check = time.time()
                return self.news_cache
                
        except Exception as e:
            logger.debug(f"News fetch error: {e}")
            self.last_news_check = time.time()
            return self.news_cache

    def analyze_sentiment(self, pair: str) -> Dict:
        """Analyze sentiment with cached data awareness"""
        if time.time() - self.last_news_check > NEWS_CHECK_INTERVAL:
            self.fetch_news()

        base_curr, quote_curr = pair.split('/')
        positive_count = 0
        negative_count = 0
        relevant_news = []

        for article in self.news_cache:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"

            if base_curr.lower() in content or quote_curr.lower() in content:
                sentiment = article.get('sentiment', 'neutral')

                if sentiment == 'positive':
                    positive_count += 1
                elif sentiment == 'negative':
                    negative_count += 1

                relevant_news.append({
                    'title': article.get('title'),
                    'sentiment': sentiment,
                    'published': article.get('published_at'),
                    'source': article.get('source')
                })

        total = positive_count + negative_count
        sentiment_score = (positive_count - negative_count) / total if total > 0 else 0.0

        return {
            'score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'relevant_news': relevant_news[:3],
            'total_news': len(self.news_cache)
        }

    def check_high_impact_news(self) -> List[Dict]:
        high_impact_keywords = [
            'central bank', 'interest rate', 'fed', 'ecb', 'boe',
            'gdp', 'employment', 'inflation', 'nonfarm payroll',
            'crisis', 'recession', 'emergency', 'breaking'
        ]

        high_impact = []

        for article in self.news_cache:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"

            published = article.get('published_at', '')
            try:
                pub_time = datetime.fromisoformat(published.replace('Z', '+00:00'))
                age_hours = (datetime.now(timezone.utc) - pub_time).total_seconds() / 3600

                if age_hours < 1:
                    for keyword in high_impact_keywords:
                        if keyword in content:
                            high_impact.append({
                                'title': article.get('title'),
                                'published': published,
                                'keyword': keyword
                            })
                            break
            except:
                pass

        self.high_impact_events = high_impact
        return high_impact

# ============================================================================
# ECONOMIC CALENDAR (WITH STRICT API RATE LIMITING) - FIXED
# ============================================================================
class EconomicCalendar:
    """Tracks economic events and their impact with strict API rate limiting"""

    def __init__(self):
        self.events = []
        self.last_check = 0

    def fetch_calendar(self, news_analyzer: NewsAnalyzer) -> List[Dict]:
        """Fetch calendar with strict rate limiting"""
        if not MARKETAUX_API_KEY:
            return self.events

        can_call, reason = api_limiter.can_make_call('marketaux')
        if not can_call:
            logger.debug(f"Marketaux calendar blocked: {reason}")
            return self.events

        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                'api_token': MARKETAUX_API_KEY,
                'symbols': 'USD,EUR,GBP,JPY',
                'filter_entities': 'true',
                'language': 'en',
                'limit': 3,
                'search': 'economic OR data OR report'
            }

            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])

                events = []
                for article in articles:
                    if any(word in article.get('title', '').lower()
                           for word in ['data', 'report', 'forecast', 'gdp', 'inflation', 'employment']):
                        events.append({
                            'title': article.get('title'),
                            'time': article.get('published_at'),
                            'impact': 'high' if any(word in article.get('title', '').lower()
                                                   for word in ['gdp', 'employment', 'rate']) else 'medium'
                        })

                self.events = events
                self.last_check = time.time()
                api_limiter.record_call('marketaux', True)
                logger.info(f"📅 Fetched {len(events)} economic events")
                return events
            elif response.status_code in [402, 429]:
                api_limiter.record_call('marketaux', False)
                self.last_check = time.time()
                return self.events
            else:
                api_limiter.record_call('marketaux', False)
                self.last_check = time.time()
                return self.events
                
        except Exception as e:
            logger.debug(f"Calendar fetch error: {e}")
            self.last_check = time.time()
            return self.events

    def get_upcoming_events(self, news_analyzer: NewsAnalyzer = None, hours_ahead: int = 2) -> List[Dict]:
        """FIXED: Now accepts news_analyzer parameter"""
        if news_analyzer and time.time() - self.last_check > CALENDAR_CHECK_INTERVAL:
            self.fetch_calendar(news_analyzer)

        now = datetime.now(timezone.utc)
        upcoming = []

        for event in self.events:
            try:
                event_time = datetime.fromisoformat(event['time'].replace('Z', '+00:00'))
                hours_until = (event_time - now).total_seconds() / 3600

                if 0 < hours_until <= hours_ahead:
                    upcoming.append(event)
            except:
                pass

        return upcoming

    def should_avoid_trading(self) -> Tuple[bool, str]:
        upcoming = self.get_upcoming_events(hours_ahead=1)
        high_impact = [e for e in upcoming if e.get('impact') == 'high']

        if high_impact:
            return True, f"High impact event in 1h: {high_impact[0]['title']}"

        return False, ""

# ============================================================================
# LEARNING MEMORY SYSTEM
# ============================================================================
class LearningMemory:
    """Persistent learning from past trades"""

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
            'successful_patterns': {},
            'failed_patterns': {},
            'pair_performance': {pair: {'wins': 0, 'losses': 0} for pair in PAIRS},
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
        with open(LEARNING_MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def save_trade_history(self):
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(self.trade_history[-1000:], f, indent=2)

    def record_trade(self, signal: Dict, outcome: str, pips: float):
        trade = {
            'pair': signal['pair'],
            'direction': signal['direction'],
            'strategy': signal['strategy'],
            'confidence': signal['confidence'],
            'outcome': outcome,
            'pips': pips,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'indicators': signal.get('indicators', {}),
            'patterns': signal.get('patterns', [])
        }

        self.trade_history.append(trade)

        pair = signal['pair']
        if outcome == 'TP_HIT':
            self.memory['pair_performance'][pair]['wins'] += 1
        else:
            self.memory['pair_performance'][pair]['losses'] += 1

        strategy = signal['strategy']
        if strategy not in self.memory['strategy_performance']:
            self.memory['strategy_performance'][strategy] = {'wins': 0, 'losses': 0, 'total_pips': 0}

        if outcome == 'TP_HIT':
            self.memory['strategy_performance'][strategy]['wins'] += 1
        else:
            self.memory['strategy_performance'][strategy]['losses'] += 1

        self.memory['strategy_performance'][strategy]['total_pips'] += pips

        for pattern in signal.get('patterns', []):
            if outcome == 'TP_HIT':
                self.memory['successful_patterns'][pattern] = \
                    self.memory['successful_patterns'].get(pattern, 0) + 1
            else:
                self.memory['failed_patterns'][pattern] = \
                    self.memory['failed_patterns'].get(pattern, 0) + 1

        self.adjust_parameters()
        self.save_memory()
        self.save_trade_history()

    def adjust_parameters(self):
        total_trades = len(self.trade_history)
        if total_trades < 20:
            return

        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t['outcome'] == 'TP_HIT')
        win_rate = wins / len(recent)

        if win_rate < 0.60:
            self.memory['learned_adjustments']['min_confidence'] = min(0.80, MIN_CONFIDENCE + 0.02)
        elif win_rate > 0.75:
            self.memory['learned_adjustments']['min_confidence'] = max(0.65, MIN_CONFIDENCE - 0.01)

        avg_win = np.mean([t['pips'] for t in recent if t['outcome'] == 'TP_HIT']) if wins > 0 else 0
        avg_loss = np.mean([abs(t['pips']) for t in recent if t['outcome'] == 'SL_HIT']) if len(recent) > wins else 0

        if avg_loss > 0 and avg_win / avg_loss < 1.5:
            self.memory['learned_adjustments']['atr_tp_mult'] = min(3.5, ATR_TP_MULT + 0.1)

    def get_pair_confidence_modifier(self, pair: str) -> float:
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

    def get_strategy_confidence_modifier(self, strategy: str) -> float:
        perf = self.memory['strategy_performance'].get(strategy, {'wins': 0, 'losses': 0})
        total = perf['wins'] + perf['losses']

        if total < 5:
            return 1.0

        win_rate = perf['wins'] / total

        if win_rate > 0.70:
            return 1.08
        elif win_rate < 0.55:
            return 0.92

        return 1.0

# ============================================================================
# SIGNAL GENERATION ENGINE
# ============================================================================
def generate_signal(pair: str, news_analyzer: NewsAnalyzer,
                   calendar: EconomicCalendar, memory: LearningMemory) -> Optional[Dict]:
    """Generate trading signal with FULL technical analysis"""

    avoid, reason = calendar.should_avoid_trading()
    if avoid:
        logger.info(f"⚠️ Avoiding {pair}: {reason}")
        return None

    high_impact = news_analyzer.check_high_impact_news()
    if high_impact:
        logger.info(f"⚠️ High impact news detected, pausing signals")
        return None

    sentiment = news_analyzer.analyze_sentiment(pair)
    df = fetch_historical_data(pair, period="5y", interval="1d")

    if len(df) < 200:
        logger.warning(f"⚠️ Insufficient data for {pair}: {len(df)} candles")
        return None

    try:
        indicators = TechnicalIndicators()
        close = df['close']

        ema12 = indicators.ema(close, 12).iloc[-1]
        ema26 = indicators.ema(close, 26).iloc[-1]
        ema50 = indicators.ema(close, 50).iloc[-1]
        ema200 = indicators.ema(close, 200).iloc[-1]

        macd_line, signal_line, histogram = indicators.macd(close)
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
        macd_hist = histogram.iloc[-1]

        adx_value, plus_di, minus_di = indicators.adx(df)
        trend_strong = adx_value.iloc[-1] > 25

        rsi = indicators.rsi(close).iloc[-1]
        stoch_k, stoch_d = indicators.stochastic(df)

        bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(close)
        current_price_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

        bullish_score = 0
        bearish_score = 0
        factors = []

        if ema12 > ema26 > ema50:
            bullish_score += 25
            factors.append("Strong uptrend (EMA 12>26>50)")
        elif ema12 < ema26 < ema50:
            bearish_score += 25
            factors.append("Strong downtrend (EMA 12<26<50)")

        if close.iloc[-1] > ema200:
            bullish_score += 15
            factors.append("Above 200 EMA (long-term bull)")
        elif close.iloc[-1] < ema200:
            bearish_score += 15
            factors.append("Below 200 EMA (long-term bear)")

        if 30 < rsi < 50:
            bullish_score += 15
            factors.append(f"RSI bullish zone ({rsi:.1f})")
        elif 50 < rsi < 70:
            bearish_score += 15
            factors.append(f"RSI bearish zone ({rsi:.1f})")

        if stoch_k.iloc[-1] < 30 and stoch_k.iloc[-1] > stoch_d.iloc[-1]:
            bullish_score += 10
            factors.append("Stochastic bullish crossover")
        elif stoch_k.iloc[-1] > 70 and stoch_k.iloc[-1] < stoch_d.iloc[-1]:
            bearish_score += 10
            factors.append("Stochastic bearish crossover")

        if macd_bullish and macd_hist > 0:
            bullish_score += 5
        elif not macd_bullish and macd_hist < 0:
            bearish_score += 5

        if current_price_position < 0.3:
            bullish_score += 20
            factors.append("Price near lower Bollinger (reversal)")
        elif current_price_position > 0.7:
            bearish_score += 20
            factors.append("Price near upper Bollinger (reversal)")

        if trend_strong:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                bullish_score += 10
                factors.append(f"ADX trend strength ({adx_value.iloc[-1]:.1f})")
            else:
                bearish_score += 10
                factors.append(f"ADX trend strength ({adx_value.iloc[-1]:.1f})")

        if bullish_score > bearish_score + 20:
            direction = 'BUY'
            base_confidence = MIN_CONFIDENCE + (bullish_score - bearish_score) / 150
        elif bearish_score > bullish_score + 20:
            direction = 'SELL'
            base_confidence = MIN_CONFIDENCE + (bearish_score - bullish_score) / 150
        else:
            return None

        if direction == 'BUY' and sentiment['score'] < -0.5:
            logger.info(f"❌ Rejecting BUY {pair} due to negative sentiment ({sentiment['score']:.2f})")
            return None
        elif direction == 'SELL' and sentiment['score'] > 0.5:
            logger.info(f"❌ Rejecting SELL {pair} due to positive sentiment ({sentiment['score']:.2f})")
            return None

        pair_modifier = memory.get_pair_confidence_modifier(pair)

        if current_price_position < 0.3 or current_price_position > 0.7:
            strategy = 'MEAN_REVERSION'
        elif trend_strong:
            strategy = 'TREND_FOLLOWING'
        else:
            strategy = 'BREAKOUT'

        strategy_modifier = memory.get_strategy_confidence_modifier(strategy)
        adjusted_confidence = base_confidence * pair_modifier * strategy_modifier
        learned_min_conf = memory.memory['learned_adjustments']['min_confidence']

        if adjusted_confidence < learned_min_conf:
            logger.debug(f"Signal rejected: confidence {adjusted_confidence:.2f} < {learned_min_conf:.2f}")
            return None

        current_price = fetch_live_price(pair)
        atr = calculate_atr(df)
        learned_sl_mult = memory.memory['learned_adjustments']['atr_sl_mult']
        learned_tp_mult = memory.memory['learned_adjustments']['atr_tp_mult']

        if direction == 'BUY':
            sl = current_price - (atr * learned_sl_mult)
            tp = current_price + (atr * learned_tp_mult)
        else:
            sl = current_price + (atr * learned_sl_mult)
            tp = current_price - (atr * learned_tp_mult)

        return {
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
                'macd_histogram': float(macd_hist),
                'stochastic_k': float(stoch_k.iloc[-1]),
                'bb_position': float(current_price_position),
                'ema12': float(ema12),
                'ema26': float(ema26),
                'ema200': float(ema200)
            },
            'patterns': [],
            'confidence_factors': factors,
            'scores': {
                'bullish': bullish_score,
                'bearish': bearish_score,
                'edge': abs(bullish_score - bearish_score)
            },
            'sentiment': sentiment,
            'news_context': {
                'relevant_news_count': len(sentiment['relevant_news']),
                'sentiment_score': sentiment['score']
            }
        }

    except Exception as e:
        logger.error(f"❌ Error generating signal for {pair}: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================================
# SIGNAL MANAGER
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
    sentiment: Dict = field(default_factory=dict)
    news_context: Dict = field(default_factory=dict)
    outcome: Optional[str] = None
    outcome_time: Optional[str] = None
    outcome_pips: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class SignalManager:
    """Manages active and archived signals"""

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
        with open(ACTIVE_SIGNALS_FILE, 'w') as f:
            json.dump([s.to_dict() for s in self.active_signals], f, indent=2)

    def broadcast_signal(self, signal_data: Dict, memory: LearningMemory) -> Signal:
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
            sentiment=signal_data.get('sentiment', {}),
            news_context=signal_data.get('news_context', {})
        )

        self.active_signals.append(signal)
        self.save_state()

        pip_multiplier = 100 if 'JPY' in signal.pair else 10000
        
        risk_pips = abs(signal.entry_price - signal.sl) * pip_multiplier
        reward_pips = abs(signal.tp - signal.entry_price) * pip_multiplier
        rr = reward_pips / risk_pips if risk_pips > 0 else 0

        logger.info(f"🎯 SIGNAL: {signal.direction} {signal.pair} @ {signal.entry_price:.5f}")
        logger.info(f"   Confidence: {signal.confidence*100:.0f}% | R:R 1:{rr:.2f} | Strategy: {signal.strategy}")
        logger.info(f"   SL: {signal.sl:.5f} (-{risk_pips:.1f} pips) | TP: {signal.tp:.5f} (+{reward_pips:.1f} pips)")

        return signal

    def update_signal_outcomes(self, memory: LearningMemory):
        """Check signal outcomes and archive completed ones"""
        now = datetime.now(timezone.utc)
        archived = []

        for signal in self.active_signals:
            expires_at = datetime.fromisoformat(signal.expires_at.replace('Z', '+00:00'))

            if now >= expires_at:
                signal.status = 'EXPIRED'
                signal.outcome = 'EXPIRED'
                signal.outcome_time = now.isoformat()
                archived.append(signal)
                continue

            try:
                current_price = fetch_live_price(signal.pair)
                
                pip_multiplier = 100 if 'JPY' in signal.pair else 10000

                if signal.direction == 'BUY':
                    if current_price >= signal.tp:
                        signal.status = 'TP_HIT'
                        signal.outcome = 'TP_HIT'
                        signal.outcome_pips = (current_price - signal.entry_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'TP_HIT', signal.outcome_pips)
                        logger.info(f"✅ TP HIT: {signal.pair} (+{signal.outcome_pips:.1f} pips)")
                    elif current_price <= signal.sl:
                        signal.status = 'SL_HIT'
                        signal.outcome = 'SL_HIT'
                        signal.outcome_pips = (current_price - signal.entry_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'SL_HIT', signal.outcome_pips)
                        logger.info(f"❌ SL HIT: {signal.pair} ({signal.outcome_pips:.1f} pips)")
                else:
                    if current_price <= signal.tp:
                        signal.status = 'TP_HIT'
                        signal.outcome = 'TP_HIT'
                        signal.outcome_pips = (signal.entry_price - current_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'TP_HIT', signal.outcome_pips)
                        logger.info(f"✅ TP HIT: {signal.pair} (+{signal.outcome_pips:.1f} pips)")
                    elif current_price >= signal.sl:
                        signal.status = 'SL_HIT'
                        signal.outcome = 'SL_HIT'
                        signal.outcome_pips = (signal.entry_price - current_price) * pip_multiplier
                        signal.outcome_time = now.isoformat()
                        archived.append(signal)
                        memory.record_trade(signal.to_dict(), 'SL_HIT', signal.outcome_pips)
                        logger.info(f"❌ SL HIT: {signal.pair} ({signal.outcome_pips:.1f} pips)")
            except Exception as e:
                logger.debug(f"Error checking signal outcome: {e}")

        for signal in archived:
            self.active_signals.remove(signal)
            self.archived_signals.append(signal)

        if archived:
            self.save_state()

    def can_broadcast_new_signal(self) -> bool:
        return len(self.active_signals) < MAX_ACTIVE_SIGNALS

    def get_stats(self) -> Dict:
        completed = [s for s in self.archived_signals if s.outcome in ['TP_HIT', 'SL_HIT']]
        total = len(completed)
        wins = sum(1 for s in completed if s.outcome == 'TP_HIT')

        total_pips = sum(s.outcome_pips for s in completed)
        win_pips = sum(s.outcome_pips for s in completed if s.outcome == 'TP_HIT')
        loss_pips = sum(abs(s.outcome_pips) for s in completed if s.outcome == 'SL_HIT')

        return {
            'total_signals': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pips': total_pips,
            'avg_win': (win_pips / wins) if wins > 0 else 0,
            'avg_loss': (loss_pips / (total - wins)) if (total - wins) > 0 else 0,
            'profit_factor': (win_pips / loss_pips) if loss_pips > 0 else 0,
            'active_signals': len(self.active_signals)
        }

# ============================================================================
# DASHBOARD STATE MANAGER - FIXED
# ============================================================================
def save_dashboard_state(controller, signal_manager, news_analyzer, calendar, memory):
    """Save complete state for dashboard"""

    uptime_hours = 0
    if controller.start_time and controller.is_running:
        start = datetime.fromisoformat(controller.start_time.replace('Z', '+00:00'))
        uptime_hours = (datetime.now(timezone.utc) - start).total_seconds() / 3600

    state = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'system_running': controller.is_running,
        'uptime_hours': uptime_hours,
        'news_count': len(news_analyzer.news_cache),
        'upcoming_events': len(calendar.get_upcoming_events(news_analyzer)),
        'high_impact_news': len(news_analyzer.high_impact_events),
        'active_signals': len(signal_manager.active_signals),
        'recent_news': news_analyzer.news_cache[:10],
        'upcoming_events_detail': calendar.get_upcoming_events(news_analyzer),
        'signals': [s.to_dict() for s in signal_manager.active_signals],
        'learning_stats': {
            'total_trades': len(memory.trade_history),
            'pair_performance': memory.memory['pair_performance'],
            'strategy_performance': memory.memory['strategy_performance'],
            'learned_params': memory.memory['learned_adjustments']
        },
        'stats': signal_manager.get_stats(),
        'price_rotation_stats': price_rotation.get_stats(),
        'api_usage': api_limiter.get_stats()
    }

    with open(DASHBOARD_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ============================================================================
# MAIN SYSTEM LOOP
# ============================================================================
def main():
    """Main system loop - Runs for 4.5 hours then exits"""
    logger.info("=" * 80)
    logger.info("🎯 TRADE BEACON - STRICT API LIMITS MODE")
    logger.info("=" * 80)
    logger.info(f"📊 Pairs: {', '.join(PAIRS)}")
    logger.info(f"🎯 Max signals: {MAX_ACTIVE_SIGNALS}")
    logger.info(f"📈 Min confidence: {MIN_CONFIDENCE*100:.0f}%")
    logger.info(f"⏰ Run Duration: 4.5 hours")
    logger.info(f"📅 Active: Monday-Friday only")
    logger.info("=" * 80)
    logger.info(api_limiter.get_summary())
    logger.info("=" * 80)

    now = datetime.now(timezone.utc)
    
    if now.weekday() in [5, 6]:
        logger.info("=" * 80)
        logger.info("🏖️  WEEKEND DETECTED - Markets Closed")
        logger.info("⏸️  System will not run. Next cycle: Monday")
        logger.info("=" * 80)
        return
    
    holidays = [(1, 1), (12, 25), (12, 26), (7, 4), (11, 11)]
    
    if (now.month, now.day) in holidays:
        logger.info("=" * 80)
        logger.info(f"🎉 HOLIDAY DETECTED - {now.strftime('%B %d')}")
        logger.info("⏸️  Markets closed. System will not run.")
        logger.info("=" * 80)
        return

    RUN_DURATION = 4.5 * 60 * 60
    start_time = time.time()
    end_time = start_time + RUN_DURATION

    controller = SystemController()
    news_analyzer = NewsAnalyzer()
    calendar = EconomicCalendar()
    memory = LearningMemory()
    signal_manager = SignalManager()

    if IN_GHA:
        controller.start()
        logger.info("✅ System AUTO-STARTED in GitHub Actions")
    else:
        logger.info("✅ System initialized - waiting for dashboard START")

    logger.info(f"⏰ Will run until: {datetime.fromtimestamp(end_time, timezone.utc).strftime('%H:%M UTC')}")
    logger.info("📱 Dashboard: signal_state/dashboard_state.json")

    last_signal_check = time.time()
    signal_pair_index = 0
    cycle_count = 0
    last_api_stats_display = time.time()

    while True:
        current_time = time.time()
        elapsed_hours = (current_time - start_time) / 3600
        remaining_minutes = (end_time - current_time) / 60

        if current_time >= end_time:
            logger.info("=" * 80)
            logger.info(f"⏰ 4.5-HOUR CYCLE COMPLETE!")
            logger.info(f"🔄 Processed {cycle_count} scan cycles")
            logger.info(f"📊 Active signals: {len(signal_manager.active_signals)}")
            stats = signal_manager.get_stats()
            logger.info(f"📈 Win Rate: {stats['win_rate']:.1f}% | Total Pips: {stats['total_pips']:.1f}")
            logger.info(f"🏆 Wins: {stats['wins']} | Losses: {stats['losses']}")
            logger.info("=" * 80)
            logger.info(api_limiter.get_summary())
            logger.info("=" * 80)
            logger.info("🛑 Exiting... Next cycle starts in ~1.5 hours")
            logger.info("=" * 80)
            
            controller.stop()
            save_dashboard_state(controller, signal_manager, news_analyzer, calendar, memory)
            break

        # Display API stats every 30 minutes
        if current_time - last_api_stats_display >= 1800:
            logger.info("\n" + "=" * 70)
            logger.info(api_limiter.get_summary())
            logger.info("=" * 70)
            last_api_stats_display = current_time

        if int(elapsed_hours) > int((current_time - 300 - start_time) / 3600):
            logger.info(f"⏰ Running for {elapsed_hours:.1f}hrs | {remaining_minutes:.0f}min remaining | Signals: {len(signal_manager.active_signals)}")

        controller.load_state()

        if not controller.is_running and not IN_GHA:
            logger.info("⏸️ System paused - waiting for START")
            save_dashboard_state(controller, signal_manager, news_analyzer, calendar, memory)
            time.sleep(5)
            continue

        now = datetime.now(timezone.utc)
        if now.weekday() in [5, 6]:
            logger.info("=" * 80)
            logger.info("🏖️  Weekend started during runtime - Stopping gracefully")
            logger.info("=" * 80)
            controller.stop()
            save_dashboard_state(controller, signal_manager, news_analyzer, calendar, memory)
            break

        should_run, reason = controller.check_should_run()

        if not should_run:
            logger.info(f"⏸️ {reason}")
            save_dashboard_state(controller, signal_manager, news_analyzer, calendar, memory)
            time.sleep(60)
            continue

        signal_manager.update_signal_outcomes(memory)

        current_time_mod_news = int(current_time) % NEWS_CHECK_INTERVAL
        current_time_mod_calendar = int(current_time) % CALENDAR_CHECK_INTERVAL
        
        if current_time_mod_news < 5:
            if time.time() - news_analyzer.last_news_check > NEWS_CHECK_INTERVAL:
                news_analyzer.fetch_news()

        if current_time_mod_calendar < 5:
            if time.time() - calendar.last_check > CALENDAR_CHECK_INTERVAL:
                calendar.fetch_calendar(news_analyzer)

        if (current_time - last_signal_check) >= SIGNAL_CHECK_INTERVAL:
            if signal_manager.can_broadcast_new_signal():
                pair = PAIRS[signal_pair_index % len(PAIRS)]
                signal_pair_index += 1
                cycle_count += 1

                logger.info(f"🔍 [{remaining_minutes:.0f}min left] Scanning {pair}... (Cycle #{cycle_count})")
                signal_data = generate_signal(pair, news_analyzer, calendar, memory)

                if signal_data:
                    signal_manager.broadcast_signal(signal_data, memory)
                else:
                    logger.debug(f"   No signal for {pair}")

            last_signal_check = current_time

        save_dashboard_state(controller, signal_manager, news_analyzer, calendar, memory)
        time.sleep(5)

# ======================================================
# 🚀 STARTUP CONTROL
# ======================================================
print("\n" + "=" * 70)
print("✅ TRADE BEACON LOADED SUCCESSFULLY - STRICT API LIMITS")
print("=" * 70)
print(f"📊 Monitoring {len(PAIRS)} currency pairs")
print(f"🎯 Maximum {MAX_ACTIVE_SIGNALS} concurrent signals")
print(f"🧠 Learning system enabled")
print(f"📰 News & calendar integration active")
print(f"🔄 Multi-source price rotation enabled")
print("=" * 70)
print("🔒 STRICT API RATE LIMITS ENFORCED:")
print(f"   • YFinance: {API_LIMITS['yfinance']['daily_limit']} calls/day")
print(f"   • Alpha Vantage: {API_LIMITS['alpha_vantage']['daily_limit']} calls/day")
print(f"   • Browserless: {API_LIMITS['browserless']['daily_limit']} calls/day (from Jan 19, 2025)")
print(f"   • Marketaux: {API_LIMITS['marketaux']['daily_limit']} calls/day")
print("=" * 70)
print(api_limiter.get_summary())
print("=" * 70)

if __name__ == "__main__":
    if IN_COLAB or IN_GHA:
        print("\n🚀 AUTO-STARTING in automated environment...")
        print("📱 Dashboard will be updated at: signal_state/dashboard_state.json")
        print("=" * 70)
        main()
    else:
        print("\n💻 LOCAL MODE: Manual start required")
        print("   Option 1: Run main() in Python")
        print("   Option 2: Use dashboard control")
        print("=" * 70)
        print("\n⏸️  Trade Beacon ready. Run main() to start.")
