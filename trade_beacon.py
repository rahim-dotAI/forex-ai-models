#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI FOREX BRAIN — PRODUCTION SIGNAL ENGINE
Optimized for scheduled execution (10-minute cycles)
"""

# =======================
# STANDARD LIBRARIES
# =======================
import os, sys, time, json, uuid, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# =======================
# THIRD-PARTY
# =======================
import yfinance as yf
import pandas as pd
import numpy as np
import holidays

# =======================
# GLOBAL SETTINGS
# =======================
RUN_INTERVAL_MINUTES = 10
MAX_ACTIVE_SIGNALS = 5
ATR_SL_MULT = 2.0
ATR_TP_MULT = 3.0
MIN_CONFIDENCE = 0.72
PRICE_CACHE_SECONDS = 600
HIST_CACHE_SECONDS = 3600

ALL_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"]

# =======================
# PATHS
# =======================
BASE = Path.cwd()
STATE = BASE / "state"
DATA = BASE / "data"
STATE.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

ACTIVE_FILE = STATE / "active_signals.json"
RISK_FILE = STATE / "risk.json"
MEMORY_FILE = STATE / "memory.json"
ANALYTICS_FILE = STATE / "analytics.json"

# =======================
# LOGGING
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# =======================
# MARKET STATUS
# =======================
def check_market_open():
    today = datetime.now(timezone.utc).date()
    if today.weekday() >= 5:
        sys.exit(0)

    us = holidays.US()
    major = ["New Year's Day", "Independence Day", "Thanksgiving", "Christmas Day"]
    if today in us and us[today] in major:
        sys.exit(0)

check_market_open()

# =======================
# SESSION LOGIC
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
# PRICE CACHE
# =======================
PRICE_CACHE = {}

def fetch_price(pair: str) -> Optional[float]:
    now = time.time()
    if pair in PRICE_CACHE:
        price, ts = PRICE_CACHE[pair]
        if now - ts < PRICE_CACHE_SECONDS:
            return price

    symbol = pair.replace("/", "") + "=X"
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    if df.empty:
        return None

    last_ts = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - last_ts).seconds > 300:
        return None

    price = float(df["Close"].iloc[-1])
    PRICE_CACHE[pair] = (price, now)
    return price

# =======================
# TECHNICALS
# =======================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    rs = up / (dn + 1e-9)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, n)
    plus_di = 100 * pd.Series(plus).rolling(n).mean() / tr
    minus_di = 100 * pd.Series(minus).rolling(n).mean() / tr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(n).mean()

# =======================
# DATA LOAD
# =======================
def load_history(pair):
    f = DATA / f"{pair.replace('/','_')}.pkl"
    if f.exists() and time.time() - f.stat().st_mtime < HIST_CACHE_SECONDS:
        return pd.read_pickle(f)
    df = yf.download(pair.replace("/","")+"=X", period="5y", auto_adjust=True, progress=False)
    if not df.empty:
        df.to_pickle(f)
    return df

# =======================
# RISK MANAGER
# =======================
class Risk:
    def __init__(self):
        self.max_dd = -500
        self.equity = 0
        self.peak = 0
        self.load()

    def load(self):
        if RISK_FILE.exists():
            d = json.loads(RISK_FILE.read_text())
            self.equity = d["equity"]
            self.peak = d["peak"]

    def save(self):
        RISK_FILE.write_text(json.dumps({"equity": self.equity, "peak": self.peak}))

    def update(self, pips):
        self.equity += pips
        self.peak = max(self.peak, self.equity)
        self.save()

    def blocked(self):
        return self.equity - self.peak <= self.max_dd

# =======================
# SIGNAL STRUCT
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
    trailing: bool = False

# =======================
# SIGNAL ENGINE
# =======================
def generate(pair, active):
    if pair in [s.pair for s in active]:
        return None

    df = load_history(pair)
    if len(df) < 200:
        return None

    close = df["Close"]
    e12, e26, e200 = ema(close,12).iloc[-1], ema(close,26).iloc[-1], ema(close,200).iloc[-1]
    r = rsi(close).iloc[-1]
    a = adx(df).iloc[-1]

    bull = bear = 0
    if e12 > e26 > e200: bull += 40
    if e12 < e26 < e200: bear += 40
    if r < 40: bull += 20
    if r > 60: bear += 20
    if a > 25: bull += 10 if e12 > e26 else 0

    if abs(bull - bear) < 30:
        return None

    side = "BUY" if bull > bear else "SELL"
    conf = min(0.85, MIN_CONFIDENCE + abs(bull-bear)/200)

    price = fetch_price(pair)
    if not price:
        return None

    a_val = atr(df).iloc[-1]
    sl = price - a_val*ATR_SL_MULT if side=="BUY" else price + a_val*ATR_SL_MULT
    tp = price + a_val*ATR_TP_MULT if side=="BUY" else price - a_val*ATR_TP_MULT

    return Signal(
        id=str(uuid.uuid4()),
        pair=pair,
        side=side,
        entry=price,
        sl=sl,
        tp=tp,
        confidence=conf,
        created=datetime.now(timezone.utc).isoformat()
    )

# =======================
# MAIN LOOP
# =======================
def main():
    session, pairs = market_session()
    log.info(f"Session: {session}")

    active = []
    if ACTIVE_FILE.exists():
        active = [Signal(**s) for s in json.loads(ACTIVE_FILE.read_text())]

    risk = Risk()
    if risk.blocked():
        log.warning("Daily drawdown limit hit")
        return

    for pair in pairs:
        if len(active) >= MAX_ACTIVE_SIGNALS:
            break
        sig = generate(pair, active)
        if sig:
            active.append(sig)
            log.info(f"NEW {sig.side} {sig.pair} @ {sig.entry:.5f}")

    ACTIVE_FILE.write_text(json.dumps([asdict(s) for s in active], indent=2))

if __name__ == "__main__":
    main()
