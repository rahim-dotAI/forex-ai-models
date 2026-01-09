import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
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
        return json.load(f)

# =========================
# CONFIG
# =========================
PAIRS = ["USDJPY=X", "AUDUSD=X", "NZDUSD=X"]
INTERVAL = "15m"
LOOKBACK = "14d"  # ✅ FIXED: Increased for stable EMA-200
MIN_ROWS = 220     # ✅ FIXED: More candles needed

CONFIG = load_config()
MODE = CONFIG["mode"]
SETTINGS = CONFIG["settings"][MODE]

SIGNAL_THRESHOLD = SETTINGS["threshold"]
MIN_ADX = SETTINGS.get("min_adx")
RSI_OVERSOLD = SETTINGS.get("rsi_oversold", 45)  # ✅ FIXED: More aggressive default
RSI_OVERBOUGHT = SETTINGS.get("rsi_overbought", 55)  # ✅ FIXED: More aggressive default

# =========================
# UTILS
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
    """Calculate Average True Range for dynamic stops"""
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
    
    # ✅ FIXED: Stars now correctly show strength (more stars = stronger signal)
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

    # ✅ IMPROVED: ATR-based dynamic stops with proper risk:reward
    if direction == "BUY":
        sl = current_price - (1.5 * atr)  # 1.5x ATR stop loss
        tp = current_price + (2.5 * atr)  # 2.5x ATR take profit (1.67:1 R:R)
    else:
        sl = current_price + (1.5 * atr)
        tp = current_price - (2.5 * atr)

    return {
        "pair": pair.replace("=X", ""),
        "direction": direction,
        "score": diff,
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
# DASHBOARD WITH TRACKING
# =========================
def write_dashboard_state(signals: list, api_calls: int):
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        session = "ASIAN"
    elif 8 <= hour < 16:
        session = "EUROPEAN"
    else:
        session = "US"

    # ✨ GET PERFORMANCE STATS ✨
    performance = track_performance(signals)

    dashboard_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "session": session,
        "mode": MODE,
        "signals": signals,
        "api_usage": {"yfinance": {"calls": api_calls}},
        "stats": performance["stats"],
        "risk_management": performance["risk_management"]
    }

    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "dashboard_state.json"
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    log.info(f"📊 Dashboard written to {output_file}")
    
    # Log performance stats
    stats = performance["stats"]
    if stats["total_trades"] > 0:
        log.info(f"📈 Performance: {stats['total_trades']} trades | "
                f"Win Rate: {stats['win_rate']}% | "
                f"Total Pips: {stats['total_pips']}")

# =========================
# TIME-WINDOW GUARD
# =========================
def in_execution_window():
    """
    Ensures the bot runs only once per execution window.
    This prevents double runs if GitHub schedule lags.
    Set to 10 minutes for a 15-minute cron schedule.
    """
    last_run_file = Path("signal_state/last_run.txt")
    now = datetime.now(timezone.utc)

    if last_run_file.exists():
        with open(last_run_file, 'r') as f:
            last_run_str = f.read().strip()
        try:
            last_run = datetime.fromisoformat(last_run_str)
            if now - last_run < timedelta(minutes=10):
                log.info(f"⏱ Already ran in the last 10 minutes ({last_run}) - exiting")
                return False
        except Exception:
            pass

    last_run_file.parent.mkdir(exist_ok=True)
    with open(last_run_file, 'w') as f:
        f.write(now.isoformat())
    return True

# =========================
# MAIN
# =========================
def main():
    if not in_execution_window():
        return

    log.info(f"🚀 Starting Trade Beacon - Mode={MODE}")
    active = []
    api_calls = 0

    for pair in PAIRS:
        log.info(f"🔍 Analyzing {pair.replace('=X','')}...")
        sig = generate_signal(pair)
        api_calls += 1
        if sig:
            active.append(sig)

    log.info(f"✅ Cycle complete | Active signals: {len(active)}")
    write_dashboard_state(active, api_calls)

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        print("\n" + "="*70)
        print(f"🎯 {MODE.upper()} SIGNALS:")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70 + "\n")
    else:
        log.info("✅ No strong signals this cycle")


if __name__ == "__main__":
    main()
