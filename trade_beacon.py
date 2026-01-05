#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI FOREX BRAIN — PRODUCTION SIGNAL ENGINE
Fully shape-safe, scalar-safe, GitHub Actions stable
"""

# =======================
# STANDARD LIBRARIES
# =======================
import sys, time, json, uuid, logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List

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
STATE.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

ACTIVE_FILE = STATE / "active_signals.json"

# =======================
# LOGGING
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
    if today in us and us[today] in {
        "New Year's Day",
        "Independence Day",
        "Thanksgiving",
        "Christmas Day",
    }:
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
# PRICE FETCH
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

    price = float(df["Close"].iloc[-1])
    PRICE_CACHE[pair] = (price, now)
    return price

# =======================
# INDICATORS (SAFE)
# =======================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"].squeeze().astype(float)
    l = df["Low"].squeeze().astype(float)
    c = df["Close"].squeeze().astype(float)

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"].squeeze().astype(float)
    l = df["Low"].squeeze().astype(float)
    c = df["Close"].squeeze().astype(float)

    up = h.diff()
    down = l.diff().abs()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_val

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

# =======================
# DATA
# =======================
def load_history(pair: str) -> pd.DataFrame:
    f = DATA / f"{pair.replace('/', '_')}.pkl"

    if f.exists() and time.time() - f.stat().st_mtime < HIST_CACHE_SECONDS:
        return pd.read_pickle(f)

    symbol = pair.replace("/", "") + "=X"
    df = yf.download(symbol, period="5y", progress=False)

    df = df[["High", "Low", "Close"]].astype(float)
    df.to_pickle(f)
    return df

# =======================
# SIGNAL
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
def generate_signal(pair: str, active: List[Signal]) -> Optional[Signal]:
    if pair in [s.pair for s in active]:
        return None

    df = load_history(pair)
    if len(df) < 200:
        return None

    close = df["Close"]

    ema12 = float(ema(close, 12).iloc[-1])
    ema26 = float(ema(close, 26).iloc[-1])
    ema200 = float(ema(close, 200).iloc[-1])
    r = float(rsi(close).iloc[-1])
    a = float(adx(df).iloc[-1])

    bull = bear = 0

    if ema12 > ema26 > ema200:
        bull += 40
    if ema12 < ema26 < ema200:
        bear += 40

    if r < 40:
        bull += 20
    if r > 60:
        bear += 20

    if a > 25:
        bull += 10 if ema12 > ema26 else 0
        bear += 10 if ema12 < ema26 else 0

    if abs(bull - bear) < 30:
        return None

    side = "BUY" if bull > bear else "SELL"
    confidence = min(0.85, MIN_CONFIDENCE + abs(bull - bear) / 200)

    price = fetch_price(pair)
    if price is None:
        return None

    atr_val = float(atr(df).iloc[-1])

    sl = price - atr_val * ATR_SL_MULT if side == "BUY" else price + atr_val * ATR_SL_MULT
    tp = price + atr_val * ATR_TP_MULT if side == "BUY" else price - atr_val * ATR_TP_MULT

    return Signal(
        id=str(uuid.uuid4()),
        pair=pair,
        side=side,
        entry=price,
        sl=sl,
        tp=tp,
        confidence=confidence,
        created=datetime.now(timezone.utc).isoformat(),
    )

# =======================
# MAIN
# =======================
def main():
    session, pairs = market_session()
    log.info(f"Session: {session}")

    active: List[Signal] = []
    if ACTIVE_FILE.exists():
        active = [Signal(**s) for s in json.loads(ACTIVE_FILE.read_text())]

    for pair in pairs:
        if len(active) >= MAX_ACTIVE_SIGNALS:
            break

        sig = generate_signal(pair, active)
        if sig:
            active.append(sig)
            log.info(
                f"NEW SIGNAL | {sig.side} {sig.pair} "
                f"@ {sig.entry:.5f} SL {sig.sl:.5f} TP {sig.tp:.5f}"
            )

    ACTIVE_FILE.write_text(json.dumps([asdict(s) for s in active], indent=2))

if __name__ == "__main__":
    main()
