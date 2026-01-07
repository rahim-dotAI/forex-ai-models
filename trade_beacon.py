import logging
import sys
from datetime import datetime, timezone

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

MIN_ROWS = 60        # Minimum candles required
SIGNAL_THRESHOLD = 30  # Minimum score difference to generate signal

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
    """Safely extract last value from a Series"""
    if series is None or series.empty:
        return None
    return float(series.iloc[-1])


def download(pair: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance"""
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
    """Calculate EMA indicator"""
    return EMAIndicator(series, window=period).ema_indicator()


def rsi(series, period=14):
    """Calculate RSI indicator"""
    return RSIIndicator(series, window=period).rsi()


def adx_calc(high, low, close):
    """Calculate ADX indicator"""
    return ADXIndicator(high, low, close, window=14).adx()


# =========================
# SIGNAL ENGINE
# =========================
def generate_signal(pair: str) -> dict | None:
    """Generate trading signal for a single pair"""
    df = download(pair)

    if len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} not enough candles ({len(df)}), skipping")
        return None

    try:
        # FIX: Flatten 2D arrays to 1D Series
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        
        # Calculate indicators
        e12 = last(ema(close, 12))
        e26 = last(ema(close, 26))
        e200 = last(ema(close, 200))
        r = last(rsi(close))
        a = last(adx_calc(high, low, close))
        
    except Exception as e:
        log.warning(f"⚠️ {pair} indicator calc failed: {e}")
        return None

    # Validate all indicators calculated successfully
    if None in (e12, e26, e200, r, a):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None

    # =========================
    # SCORING LOGIC
    # =========================
    bull = bear = 0

    # EMA structure (40 points)
    if e12 > e26 > e200:
        bull += 40
    elif e12 < e26 < e200:
        bear += 40

    # RSI momentum (20 points)
    if r < 40:
        bull += 20
    elif r > 60:
        bear += 20

    # ADX trend confirmation (10 points)
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

    # Filter weak signals
    if diff < SIGNAL_THRESHOLD:
        return None

    direction = "BUY" if bull > bear else "SELL"

    return {
        "pair": pair.replace("=X", ""),
        "direction": direction,
        "score": diff,
        "rsi": round(r, 1),
        "adx": round(a, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================
# MAIN
# =========================
def main():
    log.info("🚀 Starting Trade Beacon Analysis...")
    active = []

    for pair in PAIRS:
        log.info(f"🔍 Analyzing {pair.replace('=X', '')}...")
        sig = generate_signal(pair)
        if sig:
            active.append(sig)

    log.info(f"🚀 Cycle complete | Active signals: {len(active)}")

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        print("\n" + "="*60)
        print("ACTIVE SIGNALS:")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60 + "\n")
    else:
        log.info("✅ No signals meet threshold - market conditions neutral")


if __name__ == "__main__":
    main()
