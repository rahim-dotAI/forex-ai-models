"""
Performance Tracking System for Trade Beacon
Tracks signal outcomes, calculates win rate, pips, and performance stats
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from functools import wraps
import time
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


class PerformanceTracker:
    def __init__(self, history_file: str = "signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load signal history from file"""
        if not self.history_file.exists():
            return {
                "signals": [],
                "stats": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pips": 0.0,
                    "win_rate": 0.0
                },
                "daily": {}
            }
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                
            # Validate structure
            if not isinstance(data.get("signals"), list):
                raise ValueError("Invalid history structure")
            
            return data
            
        except Exception as e:
            log.error(f"Failed to load history: {e}")
            
            # Backup corrupted file
            if self.history_file.exists():
                backup = self.history_file.with_suffix('.json.bak')
                self.history_file.rename(backup)
                log.warning(f"Backed up corrupted file to {backup}")
            
            return {
                "signals": [],
                "stats": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pips": 0.0,
                    "win_rate": 0.0
                },
                "daily": {}
            }
    
    def _save_history(self):
        """Save signal history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save history: {e}")
    
    def add_signal(self, signal: Dict):
        """Add a new signal to tracking"""
        # Check if signal already exists (avoid duplicates)
        signal_id = f"{signal['pair']}_{signal['timestamp']}"
        
        existing = next(
            (s for s in self.history["signals"] if s.get("id") == signal_id),
            None
        )
        
        if existing:
            log.info(f"Signal {signal_id} already tracked")
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
            "closed_price": None
        }
        
        self.history["signals"].append(tracked_signal)
        log.info(f"✅ Added signal to tracking: {signal['pair']} {signal['direction']}")
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
    def _download_price_data(self, pair_symbol: str):
        """Download price data with retry logic"""
        df = yf.download(
            pair_symbol,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=True,
            threads=False
        )
        return df
    
    def _check_signal_outcome(self, signal: Dict):
        """Check if a signal hit TP or SL"""
        pair_symbol = signal["pair"] + "=X"
        
        try:
            # Get recent price data with retry logic
            df = self._download_price_data(pair_symbol)
            
            if df.empty:
                log.warning(f"⚠️ No data for {signal['pair']}")
                return
            
            # Use squeeze() consistently with trade_beacon.py
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            
            current_price = float(close.iloc[-1]) if len(close) > 0 else 0.0
            high_price = float(high.max()) if len(high) > 0 else 0.0
            low_price = float(low.min()) if len(low) > 0 else 0.0
            
            direction = signal["direction"]
            entry = signal["entry_price"]
            tp = signal["tp"]
            sl = signal["sl"]
            
            # Check if TP or SL was hit
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
            
            # Check if signal is too old (7 days) - mark as EXPIRED
            signal_time = datetime.fromisoformat(signal["timestamp"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - signal_time > timedelta(days=7):
                self._close_signal(signal, "EXPIRED", current_price, 0.0)
                log.info(f"⏰ EXPIRED: {signal['pair']} {direction} - Too old")
        
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
            self.history["daily"][today] = {"pips": 0.0, "trades": 0}
        
        if outcome in ["WIN", "LOSS"]:
            self.history["daily"][today]["pips"] += pips
            self.history["daily"][today]["trades"] += 1
    
    def _calculate_pips(self, pair: str, entry: float, exit: float, direction: str) -> float:
        """
        Calculate pips based on pair type with validation
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
        
        # Sanity check - unlikely to have 1000+ pip moves
        if abs(pips) > 1000:
            log.warning(f"⚠️ Suspicious pip value: {pips:.1f} for {pair} (entry={entry}, exit={exit}, direction={direction})")
        
        return pips
    
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
        
        log.info(f"📊 Stats: {len(closed_signals)} trades | Win Rate: {win_rate:.1f}% | Total Pips: {total_pips:.1f}")
    
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
            log.info(f"🧹 Cleaned up {removed} old signals")
            self._save_history()


# Standalone function for easy integration
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
    
    # Return stats for dashboard
    stats = tracker.get_stats()
    daily_pips = tracker.get_daily_pips()
    
    return {
        "stats": stats,
        "risk_management": {
            "daily_pips": round(daily_pips, 1),
            "max_drawdown": 0.0  # Can be calculated later
        }
    }
