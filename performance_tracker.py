"""
Performance Tracking System for Trade Beacon v2.0
Tracks signal outcomes, calculates win rate, pips, and performance stats

CRITICAL FIXES APPLIED:
- Only evaluates price data AFTER signal generation
- Blocks same-candle evaluation (minimum 15-minute age)
- Fixes 24-hour historical lookback bug
- Adds robust data shape handling
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from functools import wraps
import time
import pandas as pd
import yfinance as yf

log = logging.getLogger("performance-tracker")


# =========================
# RETRY DECORATOR (matching trade_beacon.py)
# =========================
def retry_with_backoff(max_retries=3, backoff_factor=5):
    """Retry decorator with exponential backoff for rate limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * backoff_factor
                            log.warning(f"⚠️ Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            log.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                            raise
                    else:
                        # Non-rate-limit error, raise immediately
                        raise
            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator


# =========================
# DATA SHAPE HELPER
# =========================
def ensure_series(data):
    """
    Robustly convert yfinance data to 1D Series
    Handles multi-column DataFrames and shape inconsistencies
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data.squeeze()


class PerformanceTracker:
    """
    Track forex signal performance with proper post-signal evaluation
    
    Key Features:
    - Minimum 15-minute age before evaluation
    - Only checks price data AFTER signal generation
    - Prevents false losses from historical data
    - Calculates accurate win rate and pip statistics
    """
    
    def __init__(self, history_file: str = "signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load_history()
        self.min_age_minutes = 15  # Match signal timeframe
    
    def _load_history(self) -> Dict:
        """Load signal history from file"""
        if not self.history_file.exists():
            return self._empty_history()
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                
            # Validate structure
            if not isinstance(data.get("signals"), list):
                raise ValueError("Invalid history structure")
            
            # Validate version compatibility
            version = data.get("version", "1.0.0")
            if version != "2.0.0":
                log.warning(f"⚠️ History version {version} != 2.0.0 - stats may be from old system")
            
            return data
            
        except Exception as e:
            log.error(f"Failed to load history: {e}")
            
            # Backup corrupted file
            if self.history_file.exists():
                backup = self.history_file.with_suffix('.json.bak')
                self.history_file.rename(backup)
                log.warning(f"Backed up corrupted file to {backup}")
            
            return self._empty_history()
    
    def _empty_history(self) -> Dict:
        """Return empty history structure"""
        return {
            "version": "2.0.0",
            "signals": [],
            "stats": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pips": 0.0,
                "win_rate": 0.0
            },
            "daily": {},
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def _save_history(self):
        """Save signal history to file"""
        try:
            self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save history: {e}")
    
    def add_signal(self, signal: Dict):
        """Add a new signal to tracking"""
        # Use backend-generated signal_id if available
        signal_id = signal.get('signal_id')
        
        # Fallback to old format if signal_id not present (backward compatibility)
        if not signal_id:
            signal_id = f"{signal['pair']}_{signal['timestamp']}"
            log.warning(f"⚠️ Signal missing signal_id, using fallback: {signal_id}")
        
        # Check if signal already exists (avoid duplicates)
        existing = next(
            (s for s in self.history["signals"] if s.get("id") == signal_id),
            None
        )
        
        if existing:
            log.debug(f"Signal {signal_id} already tracked")
            return
        
        tracked_signal = {
            "id": signal_id,
            "pair": signal["pair"],
            "direction": signal["direction"],
            "entry_price": signal["entry_price"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "score": signal["score"],
            "confidence": signal["confidence"],
            "timestamp": signal["timestamp"],
            "status": "OPEN",  # OPEN, WIN, LOSS, EXPIRED
            "outcome": None,
            "pips": 0.0,
            "closed_at": None,
            "closed_price": None,
            "risk_reward": signal.get("risk_reward", 0.0),
            # Store additional metadata
            "hold_time": signal.get("hold_time"),
            "eligible_modes": signal.get("eligible_modes", []),
            "session": signal.get("session"),
            "atr": signal.get("atr", 0.0),
            "spread": signal.get("spread", 0.0)
        }
        
        self.history["signals"].append(tracked_signal)
        log.info(f"✅ Added signal to tracking: {signal['pair']} {signal['direction']} (ID: {signal_id})")
        self._save_history()
    
    def check_signals(self):
        """Check all open signals for TP/SL hits"""
        open_signals = [s for s in self.history["signals"] if s["status"] == "OPEN"]
        
        if not open_signals:
            log.info("No open signals to check")
            return
        
        log.info(f"🔍 Checking {len(open_signals)} open signals...")
        
        for signal in open_signals:
            self._check_signal_outcome(signal)
        
        self._calculate_stats()
        self._save_history()
    
    @retry_with_backoff(max_retries=3, backoff_factor=5)
    def _download_price_data(self, pair_symbol: str, start_time: datetime):
        """
        Download price data with retry logic
        
        CRITICAL FIX: Only downloads data AFTER signal generation
        
        Args:
            pair_symbol: Ticker symbol (e.g., "USDJPY=X")
            start_time: Signal generation timestamp
        
        Returns:
            DataFrame with price data from start_time onward
        """
        df = yf.download(
            pair_symbol,
            start=start_time,  # ✅ FIX: Only from signal time forward
            interval="1m",
            progress=False,
            auto_adjust=True,
            threads=False
        )
        return df
    
    def _check_signal_outcome(self, signal: Dict):
        """
        Check if a signal hit TP or SL
        
        CRITICAL FIXES APPLIED:
        1. Minimum age requirement (15 minutes)
        2. Only evaluates price data AFTER signal generation
        3. Filters out any pre-signal data
        4. Uses robust data shape handling
        """
        pair_symbol = signal["pair"] + "=X"
        
        try:
            # ✅ FIX 1: Parse signal timestamp
            signal_time = datetime.fromisoformat(
                signal["timestamp"].replace("Z", "+00:00")
            )
            
            # ✅ FIX 2: Do NOT evaluate brand-new signals
            age_minutes = (datetime.now(timezone.utc) - signal_time).total_seconds() / 60
            
            if age_minutes < self.min_age_minutes:
                log.debug(f"⏳ {signal['pair']} too young ({age_minutes:.1f}m < {self.min_age_minutes}m), skipping")
                return
            
            # ✅ FIX 3: Download ONLY data after signal time
            df = self._download_price_data(pair_symbol, signal_time)
            
            if df.empty:
                log.warning(f"⚠️ No data for {signal['pair']} after signal time")
                return
            
            # ✅ FIX 4: Ensure we only have post-signal data
            df = df[df.index >= signal_time]
            
            if len(df) == 0:
                log.debug(f"⏳ {signal['pair']} no post-signal data yet")
                return
            
            # ✅ FIX 5: Use robust data shape handling
            close = ensure_series(df["Close"])
            high = ensure_series(df["High"])
            low = ensure_series(df["Low"])
            
            if len(close) == 0 or len(high) == 0 or len(low) == 0:
                log.warning(f"⚠️ Empty price data for {signal['pair']}")
                return
            
            current_price = float(close.iloc[-1])
            high_price = float(high.max())  # Now only checks POST-signal highs
            low_price = float(low.min())    # Now only checks POST-signal lows
            
            direction = signal["direction"]
            entry = signal["entry_price"]
            tp = signal["tp"]
            sl = signal["sl"]
            
            # Check if TP or SL was hit (in post-signal data only)
            if direction == "BUY":
                if high_price >= tp:
                    # TP Hit - WIN
                    pips = self._calculate_pips(signal['pair'], entry, tp, direction)
                    self._close_signal(signal, "WIN", tp, pips)
                    log.info(f"✅ WIN: {signal['pair']} BUY - TP hit at {tp} (+{pips:.1f} pips)")
                elif low_price <= sl:
                    # SL Hit - LOSS
                    pips = self._calculate_pips(signal['pair'], entry, sl, direction)
                    self._close_signal(signal, "LOSS", sl, pips)
                    log.info(f"❌ LOSS: {signal['pair']} BUY - SL hit at {sl} ({pips:.1f} pips)")
                else:
                    log.debug(f"📊 {signal['pair']} BUY still open (current: {current_price:.5f})")
            
            else:  # SELL
                if low_price <= tp:
                    # TP Hit - WIN
                    pips = self._calculate_pips(signal['pair'], entry, tp, direction)
                    self._close_signal(signal, "WIN", tp, pips)
                    log.info(f"✅ WIN: {signal['pair']} SELL - TP hit at {tp} (+{pips:.1f} pips)")
                elif high_price >= sl:
                    # SL Hit - LOSS
                    pips = self._calculate_pips(signal['pair'], entry, sl, direction)
                    self._close_signal(signal, "LOSS", sl, pips)
                    log.info(f"❌ LOSS: {signal['pair']} SELL - SL hit at {sl} ({pips:.1f} pips)")
                else:
                    log.debug(f"📊 {signal['pair']} SELL still open (current: {current_price:.5f})")
            
            # Check if signal is too old (7 days) - mark as EXPIRED
            if age_minutes > (7 * 24 * 60):
                self._close_signal(signal, "EXPIRED", current_price, 0.0)
                log.info(f"⏰ EXPIRED: {signal['pair']} {direction} - Too old ({age_minutes/60/24:.1f} days)")
        
        except Exception as e:
            log.error(f"Error checking {signal['pair']}: {e}")
    
    def _close_signal(self, signal: Dict, outcome: str, close_price: float, pips: float):
        """Mark a signal as closed"""
        signal["status"] = outcome
        signal["outcome"] = outcome
        signal["closed_at"] = datetime.now(timezone.utc).isoformat()
        signal["closed_price"] = close_price
        signal["pips"] = pips
        
        # Update daily stats
        today = datetime.now(timezone.utc).date().isoformat()
        if today not in self.history["daily"]:
            self.history["daily"][today] = {"pips": 0.0, "trades": 0, "wins": 0, "losses": 0}
        
        if outcome == "WIN":
            self.history["daily"][today]["pips"] += pips
            self.history["daily"][today]["trades"] += 1
            self.history["daily"][today]["wins"] += 1
        elif outcome == "LOSS":
            self.history["daily"][today]["pips"] += pips  # pips will be negative
            self.history["daily"][today]["trades"] += 1
            self.history["daily"][today]["losses"] += 1
    
    def _calculate_pips(self, pair: str, entry: float, exit: float, direction: str) -> float:
        """
        Calculate pips based on pair type with validation
        
        Args:
            pair: Currency pair (e.g., "USDJPY")
            entry: Entry price
            exit: Exit price
            direction: "BUY" or "SELL"
        
        Returns:
            Pip value (positive for profit, negative for loss)
        """
        # Validate prices
        if entry <= 0 or exit <= 0:
            log.error(f"Invalid prices: entry={entry}, exit={exit}")
            return 0.0
        
        # JPY pairs: 1 pip = 0.01
        if "JPY" in pair:
            pip_value = 0.01
        else:
            # Other pairs: 1 pip = 0.0001
            pip_value = 0.0001
        
        diff = exit - entry
        
        # For SELL, profit is when price goes DOWN
        if direction == "SELL":
            diff = -diff
        
        pips = diff / pip_value
        
        # Sanity check - unlikely to have 1000+ pip moves in short timeframe
        if abs(pips) > 1000:
            log.warning(f"⚠️ Suspicious pip value: {pips:.1f} for {pair} "
                       f"(entry={entry}, exit={exit}, direction={direction})")
        
        return round(pips, 1)
    
    def _calculate_stats(self):
        """Calculate overall statistics"""
        closed_signals = [
            s for s in self.history["signals"] 
            if s["status"] in ["WIN", "LOSS"]
        ]
        
        if not closed_signals:
            self.history["stats"] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pips": 0.0,
                "win_rate": 0.0
            }
            return
        
        winning = [s for s in closed_signals if s["status"] == "WIN"]
        losing = [s for s in closed_signals if s["status"] == "LOSS"]
        
        total_pips = sum(s["pips"] for s in closed_signals)
        win_rate = (len(winning) / len(closed_signals) * 100) if closed_signals else 0.0
        
        self.history["stats"] = {
            "total_trades": len(closed_signals),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_pips": round(total_pips, 1),
            "win_rate": round(win_rate, 1)
        }
        
        log.info(f"📊 Stats: {len(closed_signals)} trades | "
                f"Win Rate: {win_rate:.1f}% | Total Pips: {total_pips:.1f}")
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.history.get("stats", {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pips": 0.0,
            "win_rate": 0.0
        })
    
    def get_daily_pips(self) -> float:
        """Get today's pips"""
        today = datetime.now(timezone.utc).date().isoformat()
        return self.history.get("daily", {}).get(today, {}).get("pips", 0.0)
    
    def get_risk_metrics(self, current_signals: List[Dict]) -> Dict:
        """
        Calculate risk management metrics from current signals
        
        Args:
            current_signals: List of current active signals
            
        Returns:
            Dict with risk management data
        """
        if not current_signals:
            return {
                "total_risk_pips": 0.0,
                "max_drawdown": 0.0,
                "average_risk_reward": 0.0
            }
        
        total_risk_pips = 0.0
        risk_rewards = []
        
        for signal in current_signals:
            # Calculate risk in pips for each signal
            entry = signal.get("entry_price", 0)
            sl = signal.get("sl", 0)
            pair = signal.get("pair", "")
            direction = signal.get("direction", "BUY")
            
            if entry > 0 and sl > 0:
                risk_pips = abs(self._calculate_pips(pair, entry, sl, direction))
                total_risk_pips += risk_pips
            
            # Track risk-reward ratios
            rr = signal.get("risk_reward", 0)
            if rr > 0:
                risk_rewards.append(rr)
        
        # Calculate average risk-reward
        avg_rr = sum(risk_rewards) / len(risk_rewards) if risk_rewards else 0.0
        
        # Calculate max drawdown from history
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            "total_risk_pips": round(total_risk_pips, 1),
            "max_drawdown": round(max_drawdown, 1),
            "average_risk_reward": round(avg_rr, 2)
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history"""
        closed_signals = [
            s for s in self.history["signals"] 
            if s["status"] in ["WIN", "LOSS"]
        ]
        
        if not closed_signals:
            return 0.0
        
        # Sort by closed timestamp
        sorted_signals = sorted(
            closed_signals, 
            key=lambda x: x.get("closed_at", x.get("timestamp", ""))
        )
        
        # Calculate cumulative pips
        cumulative_pips = 0
        peak = 0
        max_dd = 0
        
        for signal in sorted_signals:
            cumulative_pips += signal.get("pips", 0)
            
            if cumulative_pips > peak:
                peak = cumulative_pips
            
            drawdown = peak - cumulative_pips
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def cleanup_old_signals(self, days: int = 30):
        """Remove signals older than X days"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        before_count = len(self.history["signals"])
        
        self.history["signals"] = [
            s for s in self.history["signals"]
            if datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")) > cutoff
        ]
        
        after_count = len(self.history["signals"])
        removed = before_count - after_count
        
        if removed > 0:
            log.info(f"🧹 Cleaned up {removed} old signals (>{days} days)")
            self._save_history()
    
    def reset_stats(self, backup: bool = True):
        """
        Reset all performance statistics
        
        Args:
            backup: Whether to backup existing data before reset
        """
        if backup and self.history_file.exists():
            backup_file = self.history_file.with_name(
                f"signal_history_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(self.history_file, 'r') as f:
                data = f.read()
            with open(backup_file, 'w') as f:
                f.write(data)
            log.info(f"📦 Backed up old stats to {backup_file}")
        
        self.history = self._empty_history()
        self._save_history()
        log.info("🔄 Performance stats reset")


# =========================
# STANDALONE FUNCTION
# =========================
def track_performance(signals: List[Dict]) -> Dict:
    """
    Track signals and return updated statistics
    
    Args:
        signals: List of new signals to track
    
    Returns:
        Dict with stats and risk_management data
    """
    tracker = PerformanceTracker()
    
    # Add new signals
    for signal in signals:
        tracker.add_signal(signal)
    
    # Check existing signals
    tracker.check_signals()
    
    # Cleanup old signals (keep last 30 days)
    tracker.cleanup_old_signals(days=30)
    
    # Get statistics
    stats = tracker.get_stats()
    daily_pips = tracker.get_daily_pips()
    risk_metrics = tracker.get_risk_metrics(signals)
    
    return {
        "stats": {
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0.0),
            "total_pips": stats.get("total_pips", 0.0),
            "wins": stats.get("winning_trades", 0),
            "losses": stats.get("losing_trades", 0)
        },
        "risk_management": {
            "daily_pips": round(daily_pips, 1),
            "total_risk_pips": risk_metrics["total_risk_pips"],
            "max_drawdown": risk_metrics["max_drawdown"],
            "average_risk_reward": risk_metrics["average_risk_reward"]
        }
    }


# =========================
# CLI UTILITY (OPTIONAL)
# =========================
if __name__ == "__main__":
    """
    Standalone CLI for performance tracker management
    
    Usage:
        python performance_tracker.py --check    # Check open signals
        python performance_tracker.py --stats    # Show stats
        python performance_tracker.py --reset    # Reset (with backup)
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    tracker = PerformanceTracker()
    
    if "--check" in sys.argv:
        tracker.check_signals()
    elif "--stats" in sys.argv:
        stats = tracker.get_stats()
        print("\n" + "="*50)
        print("📊 PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']}%")
        print(f"Total Pips: {stats['total_pips']}")
        print(f"Wins: {stats['winning_trades']}")
        print(f"Losses: {stats['losing_trades']}")
        print("="*50 + "\n")
    elif "--reset" in sys.argv:
        confirm = input("⚠️  Reset all stats? This will backup existing data. (yes/no): ")
        if confirm.lower() == 'yes':
            tracker.reset_stats(backup=True)
            print("✅ Stats reset complete")
        else:
            print("❌ Reset cancelled")
    else:
        print("Usage:")
        print("  python performance_tracker.py --check   # Check open signals")
        print("  python performance_tracker.py --stats   # Show statistics")
        print("  python performance_tracker.py --reset   # Reset stats (with backup)")
