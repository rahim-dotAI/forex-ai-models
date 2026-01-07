#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI FOREX BRAIN — SIGNAL + TRADE MANAGEMENT ENGINE
FIXED INDICATOR EXTRACTION (NO ZERO-SIGNAL BUG)
"""

import sys, time, json, uuid, logging, os
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

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

API_CALLS = 0

# =======================
# MARKET STATUS
# =======================
def check_market_open():
    today = datetime.now(timezone.utc).date()
    if today.weekday() >= 5:
        sys.exit(0)
    if today in holidays.US():
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

    price = float(df["Close"].dropna().values[-1])
    PRICE_CACHE[pair] = (price, now)
    return price

# =======================
# INDICATORS
# =======================
def ema(series, p):
    return series.ewm(span=p, adjust=False).mean()

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
    df = df[["High", "Low", "Close"]].dropna()
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
def last(series):
    series = series.dropna()
    return float(series.values[-1])

def generate_signal(pair, active):
    log.info(f"🔍 Analyzing {pair}...")

    if pair in [s.pair for s in active]:
        return None

    df = load_history(pair)
    if len(df) < 250:
        return None

    close = df["Close"]

    e12 = last(ema(close, 12))
    e26 = last(ema(close, 26))
    e200 = last(ema(close, 200))
    r = last(rsi(close))
    a = last(adx(df))

    bull = bear = 0
    if e12 > e26 > e200: bull += 40
    if e12 < e26 < e200: bear += 40
    if r < 40: bull += 20
    if r > 60: bear += 20
    if a > 25:
        bull += 10 if e12 > e26 else 0
        bear += 10 if e12 < e26 else 0

    log.info(f"📊 {pair} Bull={bull} Bear={bear} RSI={r:.1f} ADX={a:.1f}")

    if abs(bull - bear) < 30:
        return None

    side = "BUY" if bull > bear else "SELL"
    price = fetch_price(pair)
    if price is None:
        return None

    atr_v = last(atr(df))
    sl = price - atr_v * ATR_SL_MULT if side == "BUY" else price + atr_v * ATR_SL_MULT
    tp = price + atr_v * ATR_TP_MULT if side == "BUY" else price - atr_v * ATR_TP_MULT

    log.info(f"✅ SIGNAL: {pair} {side} @ {price:.5f}")

    return Signal(
        id=str(uuid.uuid4()),
        pair=pair,
        side=side,
        entry=price,
        sl=sl,
        tp=tp,
        confidence=min(0.85, MIN_CONFIDENCE + abs(bull - bear) / 200),
        created=datetime.now(timezone.utc).isoformat()
    )

# =======================
# MAIN
# =======================
def main():
    session, pairs = market_session()
    active, history = [], []

    if ACTIVE_FILE.exists():
        active = [Signal(**s) for s in json.loads(ACTIVE_FILE.read_text())]
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())

    for pair in pairs:
        if len(active) >= MAX_ACTIVE_SIGNALS:
            break
        sig = generate_signal(pair, active)
        if sig:
            active.append(sig)

    ACTIVE_FILE.write_text(json.dumps([asdict(s) for s in active], indent=2))
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

    DASHBOARD_FILE.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(active),
        "session": session
    }, indent=2))

    log.info(f"🚀 Cycle complete | Active signals: {len(active)}")

if __name__ == "__main__":
    main()
