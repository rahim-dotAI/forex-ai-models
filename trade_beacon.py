#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI FOREX BRAIN — SIGNAL + TRADE MANAGEMENT ENGINE
CRASH-FREE • NUMPY-SAFE • ADX-SAFE
"""

import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# =========================
# CONFIG
# =========================
PAIRS = ["USDJPY=X", "AUDUSD=X", "NZDUSD=X"]
INTERVAL = "15m"
LOOKBACK = "7d"

MIN_ROWS = 60        # HARD safety floor
SIGNAL_THRESHOLD = 30

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
# UTILS
# =========================
def last(series: pd.Series):
    """Safely extract last value"""
    if series is None or series.empty:
        return None
    return float(series.iloc[-1])


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


# =========================
# INDICATORS
# =========================
def ema(series, period):
    return EMAIndicator(series, period).ema_indicator()


def rsi(series):
    return RSIIndicator(series).rsi()


def adx(df):
    return ADXIndicator(df["High"], df["Low"], df["Close"]).adx()


# =========================
# SIGNAL ENGINE
# =========================
def generate_signal(pair: str) -> dict | None:
    df = download(pair)

    if len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} not enough candles ({len(df)}), skipping")
        return None

    try:
        e12 = last(ema(df["Close"], 12))
        e26 = last(ema(df["Close"], 26))
        e200 = last(ema(df["Close"], 200))
        r = last(rsi(df["Close"]))
        a = last(adx(df))
    except Exception as e:
        log.warning(f"⚠️ {pair} indicator calc failed: {e}")
        return None

    if None in (e12, e26, e200, r, a):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None

    bull = bear = 0

    # EMA structure
    if e12 > e26 > e200:
        bull += 40
    elif e12 < e26 < e200:
        bear += 40

    # RSI
    if r < 40:
        bull += 20
    elif r > 60:
        bear += 20

    # ADX confirmation
    if a > 25:
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10

    diff = abs(bull - bear)

    log.info(
        f"{pair} | Bull={bull} Bear={bear} Diff={diff} "
        f"RSI={r:.1f} ADX={a:.1f}"
    )

    if diff < SIGNAL_THRESHOLD:
        return None

    direction = "BUY" if bull > bear else "SELL"

    return {
        "pair": pair.replace("=X", ""),
        "direction": direction,
        "score": diff,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================
# MAIN
# =========================
def main():
    active = []

    for pair in PAIRS:
        log.info(f"🔍 Analyzing {pair.replace('=X','')}...")
        sig = generate_signal(pair)
        if sig:
            active.append(sig)

    log.info(f"🚀 Cycle complete | Active signals: {len(active)}")

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")


if __name__ == "__main__":
    main()
