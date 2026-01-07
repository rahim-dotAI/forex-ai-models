#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI FOREX BRAIN — SIGNAL + TRADE MANAGEMENT ENGINE
FULL PERFORMANCE TRACKING
"""

import sys, time, json, uuid, logging, os
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List

import yfinance as yf
import pandas as pd
import numpy as np
import holidays

# =======================
# SETTINGS
# =======================
MAX_ACTIVE_SIGNALS = 5
ATR_SL_MULT = 2.0
ATR_TP_MULT = 3.0
MIN_CONFIDENCE = 0.72
PRICE_CACHE_SECONDS = 600
HIST_CACHE_SECONDS = 3600

# =======================
# PATHS
# =======================
BASE = Path.cwd()
STATE = BASE / "state"
DATA = BASE / "data"
SIGNAL_STATE = BASE / "signal_state"

STATE.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
SIGNAL_STATE.mkdir(exist_ok=True)

ACTIVE_FILE = STATE / "active_signals.json"
HISTORY_FILE = STATE / "trade_history.json"
DASHBOARD_FILE = SIGNAL_STATE / "dashboard_state.json"

# =======================
# LOGGING
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# =======================
# GLOBAL COUNTERS
# =======================
API_CALLS = 0

# =======================
# MARKET STATUS
# =======================
def check_market_open():
    today = datetime.now(timezone.utc).date()
    if today.weekday() >= 5:
        sys.exit(0)
    us = holidays.US()
    if today in us:
        sys.exit(0)

check_market_open()

# =======================
# SESSION
# =======================
def market_session():
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 7:
        return "ASIAN", ["USD/JPY", "AUD/USD", "NZD/USD"]
    if 7 <= h < 12:
        return "LONDON", ["EUR/USD", "GBP/USD"]
    if 12 <= h < 16:
        return "LONDON_NY_OVERLAP", ["EUR/USD", "GBP/USD", "USD/CAD"]
    return "NEW_YORK", ["EUR/USD", "USD/CAD"]

# =======================
# PRICE FETCH
# =======================
PRICE_CACHE = {}

def fetch_price(pair: str) -> Optional[float]:
    global API_CALLS
    now = time.time()
    if pair in PRICE_CACHE:
        price, ts = PRICE_CACHE[pair]
        if now - ts < PRICE_CACHE_SECONDS:
            return price

    API_CALLS += 1
    symbol = pair.replace("/", "") + "=X"
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    if df.empty:
        return None

    price = float(df["Close"].iloc[-1])
    PRICE_CACHE[pair] = (price, now)
    return price

# =======================
# INDICATORS
# =======================
def ema(series, p): return series.ewm(span=p, adjust=False).mean()

def rsi(series, p=14):
    d = series.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    rs = g.ewm(alpha=1/p, adjust=False).mean() / (l.ewm(alpha=1/p, adjust=False).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, p=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def adx(df, p=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1/p, adjust=False).mean()
    up = h.diff()
    down = -l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    pdi = 100 * plus.ewm(alpha=1/p, adjust=False).mean() / (atr_v + 1e-9)
    mdi = 100 * minus.ewm(alpha=1/p, adjust=False).mean() / (atr_v + 1e-9)
    dx = (pdi - mdi).abs() / (pdi + mdi + 1e-9) * 100
    return dx.ewm(alpha=1/p, adjust=False).mean()

# =======================
# DATA
# =======================
def load_history(pair):
    global API_CALLS
    f = DATA / f"{pair.replace('/', '_')}.pkl"
    if f.exists() and time.time() - f.stat().st_mtime < HIST_CACHE_SECONDS:
        return pd.read_pickle(f)
    
    API_CALLS += 1
    symbol = pair.replace("/", "") + "=X"
    df = yf.download(symbol, period="5y", progress=False)
    df = df[["High", "Low", "Close"]].astype(float)
    df.to_pickle(f)
    return df

# =======================
# MODELS
# =======================
@dataclass
class Signal:
    id: str
    pair: str
    side: str
    entry: float
    sl: float
    tp: float
    confidence: float
    created: str

# =======================
# SIGNAL ENGINE
# =======================
def generate_signal(pair, active):
    log.info(f"🔍 Analyzing {pair}...")
    
    if pair in [s.pair for s in active]:
        log.info(f"  ⏭️  {pair} already active, skipping")
        return None

    df = load_history(pair)
    if len(df) < 200:
        log.warning(f"  ❌ {pair} insufficient data: {len(df)} bars")
        return None

    close = df["Close"]
    e12 = ema(close, 12).iloc[-1]
    e26 = ema(close, 26).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    r = rsi(close).iloc[-1]
    a = adx(df).iloc[-1]
    
    # Convert to native Python types using .item() for pandas compatibility
    try:
        e12 = e12.item() if hasattr(e12, 'item') else float(e12)
        e26 = e26.item() if hasattr(e26, 'item') else float(e26)
        e200 = e200.item() if hasattr(e200, 'item') else float(e200)
        r = r.item() if hasattr(r, 'item') else float(r)
        a = a.item() if hasattr(a, 'item') else float(a)
    except (ValueError, AttributeError) as e:
        log.warning(f"{pair} ❌ Failed to convert indicators: {e}")
        return None

    bull = bear = 0
    if e12 > e26 > e200: bull += 40
    if e12 < e26 < e200: bear += 40
    if r < 40: bull += 20
    if r > 60: bear += 20
    if a > 25:
        bull += 10 if e12 > e26 else 0
        bear += 10 if e12 < e26 else 0

    # Log signal analysis for debugging
    log.info(f"  📊 {pair}: Bull={bull} Bear={bear} Diff={abs(bull-bear)} | RSI={r:.1f} ADX={a:.1f}")

    if abs(bull - bear) < 30:
        log.info(f"  ⚠️  {pair} signal too weak (need 30+ diff)")
        return None

    side = "BUY" if bull > bear else "SELL"
    price = fetch_price(pair)
    if price is None:
        log.warning(f"  ❌ {pair} failed to fetch price")
        return None

    atr_v = atr(df).iloc[-1]
    
    # Convert ATR to scalar using helper function
    def to_scalar(val):
        if hasattr(val, 'item'):
            return val.item()
        elif hasattr(val, '__iter__') and not isinstance(val, str):
            return float(list(val)[0]) if len(list(val)) > 0 else float(val)
        else:
            return float(val)
    
    try:
        atr_v = to_scalar(atr_v)
    except (ValueError, AttributeError, TypeError) as e:
        log.warning(f"  ❌ {pair} failed to convert ATR: {e}")
        return None
    
    sl = price - atr_v * ATR_SL_MULT if side == "BUY" else price + atr_v * ATR_SL_MULT
    tp = price + atr_v * ATR_TP_MULT if side == "BUY" else price - atr_v * ATR_TP_MULT

    log.info(f"  ✅ {pair} SIGNAL GENERATED: {side} @ {price:.5f}")

    return Signal(
        id=str(uuid.uuid4()),
        pair=pair,
        side=side,
        entry=price,
        sl=sl,
        tp=tp,
        confidence=min(0.85, MIN_CONFIDENCE + abs(bull - bear)/200),
        created=datetime.now(timezone.utc).isoformat()
    )

# =======================
# TRADE MANAGEMENT
# =======================
def pips(pair, entry, exit):
    factor = 0.01 if "JPY" in pair else 0.0001
    return (exit - entry) / factor

# =======================
# MAIN
# =======================
def main():
    session, pairs = market_session()
    active = []
    history = []

    if ACTIVE_FILE.exists():
        active = [Signal(**s) for s in json.loads(ACTIVE_FILE.read_text())]
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())

    still_active = []
    for s in active:
        price = fetch_price(s.pair)
        if price is None:
            still_active.append(s)
            continue

        hit_tp = price >= s.tp if s.side == "BUY" else price <= s.tp
        hit_sl = price <= s.sl if s.side == "BUY" else price >= s.sl

        if hit_tp or hit_sl:
            exit_price = s.tp if hit_tp else s.sl
            pip_val = pips(s.pair, s.entry, exit_price)
            if s.side == "SELL":
                pip_val *= -1

            history.append({
                "pair": s.pair,
                "side": s.side,
                "entry": s.entry,
                "exit": exit_price,
                "pips": round(pip_val, 1),
                "result": "WIN" if hit_tp else "LOSS",
                "closed_at": datetime.now(timezone.utc).isoformat()
            })
        else:
            still_active.append(s)

    active = still_active

    for pair in pairs:
        if len(active) >= MAX_ACTIVE_SIGNALS:
            break
        sig = generate_signal(pair, active)
        if sig:
            active.append(sig)

    ACTIVE_FILE.write_text(json.dumps([asdict(s) for s in active], indent=2))
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

    wins = [t for t in history if t["result"] == "WIN"]
    losses = [t for t in history if t["result"] == "LOSS"]
    total = len(wins) + len(losses)

    today = date.today().isoformat()
    daily_pips = sum(t["pips"] for t in history if t["closed_at"].startswith(today))

    # Calculate next run time (10 minutes from now)
    next_run = datetime.now(timezone.utc) + timedelta(minutes=10)

    dashboard = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "next_run": next_run.isoformat(),
        "active_signals": len(active),
        "signals": [{
            "pair": s.pair,
            "direction": s.side,
            "entry_price": s.entry,
            "sl": s.sl,
            "tp": s.tp
        } for s in active],
        "stats": {
            "win_rate": round((len(wins) / total * 100) if total else 0.0, 1),
            "total_pips": round(sum(t["pips"] for t in history), 1)
        },
        "risk_management": {
            "daily_pips": round(daily_pips, 1)
        },
        "api_usage": {
            "yfinance": {
                "calls": API_CALLS
            }
        },
        "metadata": {
            "session": session,
            "run_mode": os.getenv("GITHUB_EVENT_NAME", "local"),
            "pairs_checked": pairs,
            "pairs_analyzed": len(pairs)
        }
    }

    DASHBOARD_FILE.write_text(json.dumps(dashboard, indent=2))
    log.info(f"Trade Beacon cycle complete | Session: {session} | API Calls: {API_CALLS} | Active: {len(active)}")

if __name__ == "__main__":
    main()
