import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# =========================
# LOAD DYNAMIC CONFIG
# =========================
def load_config():
    """Load trading mode from config.json"""
    config_path = Path("config.json")
    if not config_path.exists():
        log.warning("⚠️ config.json not found, using aggressive defaults")
        return {
            "mode": "aggressive",
            "settings": {
                "aggressive": {
                    "threshold": 30,
                    "min_adx": None,
                    "rsi_oversold": 40,
                    "rsi_overbought": 60
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
LOOKBACK = "7d"
MIN_ROWS = 60

# Load mode settings
CONFIG = load_config()
MODE = CONFIG["mode"]
SETTINGS = CONFIG["settings"][MODE]

SIGNAL_THRESHOLD = SETTINGS["threshold"]
MIN_ADX = SETTINGS.get("min_adx")
RSI_OVERSOLD = SETTINGS.get("rsi_oversold", 40)
RSI_OVERBOUGHT = SETTINGS.get("rsi_overbought", 60)

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
# SIGNAL ENGINE - DYNAMIC MODE
# =========================
def generate_signal(pair: str) -> dict | None:
    """Generate trading signal based on current mode"""
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
        
        # Get current price for entry
        current_price = last(close)
        
    except Exception as e:
        log.warning(f"⚠️ {pair} indicator calc failed: {e}")
        return None

    # Validate all indicators calculated successfully
    if None in (e12, e26, e200, r, a, current_price):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None

    # PRE-FILTER: ADX check (only in conservative mode)
    if MIN_ADX is not None and a < MIN_ADX:
        log.info(f"❌ {pair} | ADX too low ({a:.1f} < {MIN_ADX}) - no strong trend")
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

    # RSI momentum (scoring depends on mode)
    if MODE == "conservative":
        # Strict: only extreme zones get points
        if r < RSI_OVERSOLD:
            bull += 30
        elif r > RSI_OVERBOUGHT:
            bear += 30
    else:
        # Aggressive: wider zones
        if r < RSI_OVERSOLD:
            bull += 20
        elif r > RSI_OVERBOUGHT:
            bear += 20

    # ADX trend confirmation
    if a > 25:
        if e12 > e26:
            bull += 20
        elif e12 < e26:
            bear += 20
    elif MIN_ADX and a > MIN_ADX:
        # Moderate trend (only in conservative mode)
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10
    elif not MIN_ADX and a > 15:
        # Aggressive mode: accept weaker trends
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10

    diff = abs(bull - bear)

    # Quality rating
    quality = "⭐" if diff >= 70 else "⭐⭐" if diff >= 60 else "⭐⭐⭐" if diff >= 50 else ""

    log.info(
        f"{pair} | Bull={bull} Bear={bear} Diff={diff} {quality} | "
        f"RSI={r:.1f} ADX={a:.1f}"
    )

    # Filter weak signals
    if diff < SIGNAL_THRESHOLD:
        log.info(f"❌ {pair} | Signal too weak (diff={diff} < {SIGNAL_THRESHOLD})")
        return None

    direction = "BUY" if bull > bear else "SELL"
    
    # Confidence classification
    if diff >= 70:
        confidence = "EXCELLENT"
    elif diff >= 60:
        confidence = "STRONG"
    elif diff >= 50:
        confidence = "GOOD"
    else:
        confidence = "MODERATE"

    # Calculate SL and TP based on direction
    if direction == "BUY":
        sl = current_price * 0.985  # 1.5% stop loss
        tp = current_price * 1.020  # 2% take profit
    else:
        sl = current_price * 1.015  # 1.5% stop loss
        tp = current_price * 0.980  # 2% take profit

    return {
        "pair": pair.replace("=X", ""),
        "direction": direction,
        "score": diff,
        "confidence": confidence,
        "rsi": round(r, 1),
        "adx": round(a, 1),
        "entry_price": round(current_price, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================
# WRITE DASHBOARD STATE
# =========================
def write_dashboard_state(signals: list, api_calls: int):
    """Write the dashboard state JSON file"""
    
    # Determine session based on current UTC hour
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        session = "ASIAN"
    elif 8 <= hour < 16:
        session = "EUROPEAN"
    else:
        session = "US"
    
    # Calculate stats (placeholder - you can enhance this)
    total_pips = 0
    win_rate = 0
    daily_pips = 0
    
    dashboard_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "session": session,
        "signals": signals,
        "stats": {
            "win_rate": win_rate,
            "total_pips": total_pips
        },
        "risk_management": {
            "daily_pips": daily_pips
        },
        "api_usage": {
            "yfinance": {
                "calls": api_calls
            }
        }
    }
    
    # Create directory if it doesn't exist
    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    
    # Write JSON file
    output_file = output_dir / "dashboard_state.json"
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    log.info(f"📊 Dashboard state written to {output_file}")


# =========================
# MAIN
# =========================
def main():
    mode_label = "HIGH CONFIDENCE" if MODE == "conservative" else "AGGRESSIVE"
    log.info(f"🚀 Starting Trade Beacon Analysis - {mode_label} MODE")
    log.info(
        f"📊 Settings: Threshold={SIGNAL_THRESHOLD}, "
        f"Min ADX={MIN_ADX or 'None'}, "
        f"RSI Zones=<{RSI_OVERSOLD} or >{RSI_OVERBOUGHT}"
    )
    
    active = []
    api_calls = 0

    for pair in PAIRS:
        log.info(f"🔍 Analyzing {pair.replace('=X', '')}...")
        sig = generate_signal(pair)
        api_calls += 1  # Count each API call
        if sig:
            active.append(sig)

    log.info(f"🚀 Cycle complete | Active signals: {len(active)}")

    # Always write dashboard state (even if no signals)
    write_dashboard_state(active, api_calls)

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        print("\n" + "="*70)
        print(f"🎯 {mode_label} SIGNALS:")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70 + "\n")
    else:
        log.info(f"✅ No signals meet {mode_label.lower()} criteria - waiting for better setups")


if __name__ == "__main__":
    main()
