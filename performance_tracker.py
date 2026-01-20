"""
Performance Tracking System for Trade Beacon v2.0.4-FINAL

✅ BULLETPROOF VERSION with complete type safety
✅ All numeric fields guaranteed to be int/float
✅ Handles missing files, corrupt data, empty CSVs
✅ Never raises type comparison errors
✅ Fully compatible with current pipeline

ALIGNMENTS (v2.0.4):
- Version consistency with engine
- Signal age aligned to 10 minutes (600s)
- Numeric confidence → labeled buckets
- Advisory-only optimization (no auto mutation)
- Always returns 'stats' and 'analytics'
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, Any, Union

import pandas as pd
import yfinance as yf

log = logging.getLogger("performance-tracker")

# ==========================================================
# SAFE TYPE CONVERSION UTILITIES
# ==========================================================
def safe_int(val: Any, default: int = 0) -> int:
    """Convert any value to int safely, return default on failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_float(val: Any, default: float = 0.0) -> float:
    """Convert any value to float safely, return default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_round(val: Any, decimals: int = 1) -> float:
    """Safely round a numeric value."""
    return round(safe_float(val), decimals)

# ==========================================================
# RETRY DECORATOR
# ==========================================================
def retry_with_backoff(max_retries=3, backoff_factor=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if "rate limit" in msg or "429" in msg:
                        if attempt < max_retries - 1:
                            wait = (2 ** attempt) * backoff_factor
                            log.warning(f"⚠️ Rate limit hit, retrying in {wait}s")
                            time.sleep(wait)
                        else:
                            raise
                    else:
                        raise
            raise RuntimeError("Max retries exceeded")
        return wrapper
    return decorator

# ==========================================================
# PERFORMANCE OPTIMIZER (ADVISORY)
# ==========================================================
class PerformanceOptimizer:
    def __init__(self, tracker):
        self.tracker = tracker
        self.min_trades = 30

    def get_optimal_parameters(self) -> Dict:
        """Generate optimization recommendations (advisory only)."""
        closed = [s for s in self.tracker.history["signals"]
                  if s["status"] in ("WIN", "LOSS")]

        if len(closed) < self.min_trades:
            return self._default()

        analytics = self.tracker.get_analytics()

        return {
            "advisory_only": True,
            "recommended_sessions": self._best_sessions(analytics["by_session"]),
            "optimal_pairs": self._best_pairs(analytics["by_pair"]),
            "min_confidence": self._best_confidence(analytics["by_confidence"]),
            "risk_reward_insights": self._risk_reward(closed),
            "threshold_adjustment": self._threshold_advice(closed),
            "total_trades_analyzed": len(closed),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _best_sessions(self, data: Dict) -> list:
        """Rank sessions by win rate."""
        ranked = []
        for session, stats in data.items():
            trades = safe_int(stats.get("trades", 0))
            if trades >= 5:
                win_rate = safe_float(stats.get("win_rate", 0))
                ranked.append((session, win_rate))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:3]]

    def _best_pairs(self, data: Dict) -> list:
        """Rank pairs by win rate and pips."""
        ranked = []
        for pair, stats in data.items():
            trades = safe_int(stats.get("trades", 0))
            if trades >= 5:
                win_rate = safe_float(stats.get("win_rate", 0))
                pips = safe_float(stats.get("pips", 0))
                ranked.append((pair, win_rate, pips))
        
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [p for p, wr, _ in ranked if wr >= 55]

    def _best_confidence(self, data: Dict) -> str:
        """Determine optimal confidence threshold."""
        for level in ["EXCELLENT", "STRONG", "GOOD"]:
            stats = data.get(level, {})
            trades = safe_int(stats.get("trades", 0))
            win_rate = safe_float(stats.get("win_rate", 0))
            
            if trades >= 5 and win_rate >= 55:
                return level
        return "GOOD"

    def _risk_reward(self, signals: list) -> Dict:
        """Analyze risk/reward ratio performance."""
        buckets = {"<1.5": [], "1.5-2.0": [], "2.0+": []}
        
        for s in signals:
            rr = safe_float(s.get("risk_reward", 0))
            win = s.get("status") == "WIN"
            
            if rr < 1.5:
                buckets["<1.5"].append(win)
            elif rr < 2.0:
                buckets["1.5-2.0"].append(win)
            else:
                buckets["2.0+"].append(win)

        summary = {}
        best = None
        best_wr = 0
        
        for k, v in buckets.items():
            if not v:
                continue
            wr = sum(v) / len(v) * 100
            summary[k] = {"trades": len(v), "win_rate": safe_round(wr, 1)}
            if wr > best_wr and len(v) >= 5:
                best_wr = wr
                best = k

        return {
            "by_range": summary,
            "recommended_range": best or "1.5-2.0",
            "recommended_min_rr": 1.5 if best != "<1.5" else 2.0
        }

    def _threshold_advice(self, signals: list) -> Dict:
        """Suggest threshold adjustments based on win rate."""
        wins = sum(1 for s in signals if s.get("status") == "WIN")
        total = len(signals)
        wr = safe_float(wins / total * 100 if total else 0)

        if wr < 50:
            return {"action": "RAISE", "amount": 5, "win_rate": safe_round(wr, 1)}
        if wr > 65:
            return {"action": "LOWER", "amount": 3, "win_rate": safe_round(wr, 1)}
        return {"action": "MAINTAIN", "amount": 0, "win_rate": safe_round(wr, 1)}

    def _default(self) -> Dict:
        """Return default optimization response."""
        return {
            "advisory_only": True,
            "recommended_sessions": [],
            "optimal_pairs": [],
            "min_confidence": "GOOD",
            "risk_reward_insights": {},
            "threshold_adjustment": {"action": "MAINTAIN", "amount": 0, "win_rate": 0.0},
            "total_trades_analyzed": 0,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }

# ==========================================================
# PERFORMANCE TRACKER
# ==========================================================
class PerformanceTracker:
    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.min_age_minutes = 10
        self.history = self._load()

    # ---------------- LOAD / SAVE ----------------
    def _load(self) -> Dict:
        """Load history with guaranteed structure."""
        if not self.history_file.exists():
            return self._empty()
        
        try:
            with open(self.history_file) as f:
                data = json.load(f)
            
            # Ensure all required keys exist
            data.setdefault("version", "2.0.4")
            data.setdefault("signals", [])
            data.setdefault("stats", self._empty_stats())
            data.setdefault("analytics", self._empty_analytics())
            data.setdefault("daily", {})
            data.setdefault("metadata", {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            })
            
            # Sanitize stats to ensure numeric types
            data["stats"] = self._sanitize_stats(data["stats"])
            
            return data
            
        except Exception as e:
            log.warning(f"⚠️ Could not load history: {e}, using empty state")
            return self._empty()

    def _empty(self) -> Dict:
        """Return empty history structure."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": "2.0.4",
            "signals": [],
            "stats": self._empty_stats(),
            "daily": {},
            "analytics": self._empty_analytics(),
            "metadata": {"created_at": now, "last_updated": now}
        }

    def _empty_stats(self) -> Dict:
        """Return empty stats with correct types."""
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pips": 0.0
        }

    def _empty_analytics(self) -> Dict:
        """Return empty analytics structure."""
        return {
            "by_pair": {},
            "by_session": {},
            "by_confidence": {}
        }

    def _sanitize_stats(self, stats: Dict) -> Dict:
        """Ensure all stats values are correct numeric types."""
        return {
            "total_trades": safe_int(stats.get("total_trades", 0)),
            "wins": safe_int(stats.get("wins", 0)),
            "losses": safe_int(stats.get("losses", 0)),
            "win_rate": safe_float(stats.get("win_rate", 0.0)),
            "total_pips": safe_float(stats.get("total_pips", 0.0))
        }

    def _save(self):
        """Save history to disk."""
        try:
            self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            # Ensure stats are sanitized before saving
            self.history["stats"] = self._sanitize_stats(self.history["stats"])
            
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            log.error(f"⚠️ Failed to save history: {e}")

    # ---------------- CONFIDENCE BUCKET ----------------
    @staticmethod
    def _confidence_bucket(score: Union[int, float, str]) -> str:
        """Convert numeric score to confidence label."""
        score_num = safe_float(score, 0)
        
        if score_num >= 85:
            return "EXCELLENT"
        if score_num >= 70:
            return "STRONG"
        if score_num >= 55:
            return "GOOD"
        return "MODERATE"

    # ---------------- SIGNAL INGEST ----------------
    def add_signal(self, signal: Dict):
        """Add a new signal to tracking history."""
        sid = signal.get("signal_id") or f"{signal['pair']}_{signal['timestamp']}"
        
        # Prevent duplicates
        if any(s.get("id") == sid for s in self.history["signals"]):
            return

        # Add with type-safe values
        self.history["signals"].append({
            "id": sid,
            "pair": str(signal.get("pair", "")),
            "direction": str(signal.get("direction", "")),
            "entry_price": safe_float(signal.get("entry_price", 0)),
            "sl": safe_float(signal.get("sl", 0)),
            "tp": safe_float(signal.get("tp", 0)),
            "risk_reward": safe_float(signal.get("risk_reward", 0)),
            "confidence_score": safe_float(signal.get("confidence", 0)),
            "confidence": self._confidence_bucket(signal.get("confidence", 0)),
            "session": str(signal.get("session", "UNKNOWN")),
            "timestamp": str(signal.get("timestamp", datetime.now(timezone.utc).isoformat())),
            "status": "OPEN",
            "pips": 0.0,
            "closed_at": None,
            "exit_index": None
        })
        self._save()

    # ---------------- PRICE CHECK ----------------
    def check_signals(self):
        """Check all open signals for TP/SL hits."""
        open_signals = [s for s in self.history["signals"] if s.get("status") == "OPEN"]
        
        for s in open_signals:
            try:
                self._check_one(s)
            except Exception as e:
                log.warning(f"⚠️ Failed to check {s.get('id')}: {e}")
        
        self._stats()
        self._analytics()
        self._save()

    @retry_with_backoff()
    def _download(self, pair: str, start: datetime):
        """Download price data with retry logic."""
        return yf.download(
            pair + "=X",
            start=start,
            interval="1m",
            progress=False,
            auto_adjust=True,
            threads=False
        )

    def _check_one(self, s: Dict):
        """Check individual signal for TP/SL."""
        ts_str = s.get("timestamp", "").replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            return
        
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age < self.min_age_minutes:
            return

        df = self._download(s.get("pair", ""), ts)
        if df.empty:
            return

        entry = safe_float(s.get("entry_price", 0))
        sl = safe_float(s.get("sl", 0))
        tp = safe_float(s.get("tp", 0))
        pip = 0.01 if "JPY" in s.get("pair", "") else 0.0001

        for idx, r in df.iterrows():
            hi = safe_float(r.get("High", 0))
            lo = safe_float(r.get("Low", 0))
            
            if s.get("direction") == "BUY":
                if lo <= sl:
                    self._close(s, "LOSS", sl, (sl - entry) / pip, idx)
                    return
                if hi >= tp:
                    self._close(s, "WIN", tp, (tp - entry) / pip, idx)
                    return
            else:  # SELL
                if hi >= sl:
                    self._close(s, "LOSS", sl, (entry - sl) / pip, idx)
                    return
                if lo <= tp:
                    self._close(s, "WIN", tp, (entry - tp) / pip, idx)
                    return

    def _close(self, s: Dict, status: str, price: float, pips: float, idx):
        """Mark signal as closed and update daily stats."""
        s["status"] = status
        s["closed_at"] = datetime.now(timezone.utc).isoformat()
        s["pips"] = safe_round(pips, 1)
        s["exit_index"] = str(idx)

        # Update daily stats
        try:
            day = pd.Timestamp(idx).date().isoformat()
        except Exception:
            day = datetime.now(timezone.utc).date().isoformat()
        
        self.history.setdefault("daily", {})
        self.history["daily"].setdefault(day, {"pips": 0.0, "trades": 0})
        self.history["daily"][day]["pips"] = safe_float(self.history["daily"][day]["pips"]) + s["pips"]
        self.history["daily"][day]["trades"] = safe_int(self.history["daily"][day]["trades"]) + 1

    # ---------------- STATS / ANALYTICS ----------------
    def _stats(self):
        """Calculate overall performance statistics."""
        closed = [s for s in self.history["signals"] 
                  if s.get("status") in ("WIN", "LOSS")]
        
        wins = sum(1 for s in closed if s.get("status") == "WIN")
        total = len(closed)
        pips = sum(safe_float(s.get("pips", 0)) for s in closed)
        
        self.history["stats"] = {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": safe_round(wins / total * 100 if total else 0, 1),
            "total_pips": safe_round(pips, 1)
        }

    def _analytics(self):
        """Generate detailed analytics by pair, session, confidence."""
        analytics = self._empty_analytics()
        
        for s in self.history["signals"]:
            if s.get("status") not in ("WIN", "LOSS"):
                continue

            # Group by pair, session, and confidence
            for key, val in [
                ("by_pair", s.get("pair", "UNKNOWN")),
                ("by_session", s.get("session", "UNKNOWN")),
                ("by_confidence", s.get("confidence", "UNKNOWN"))
            ]:
                analytics[key].setdefault(val, {
                    "wins": 0,
                    "losses": 0,
                    "pips": 0.0,
                    "trades": 0
                })
                
                d = analytics[key][val]
                d["trades"] = safe_int(d["trades"]) + 1
                d["pips"] = safe_float(d["pips"]) + safe_float(s.get("pips", 0))
                
                if s.get("status") == "WIN":
                    d["wins"] = safe_int(d["wins"]) + 1
                else:
                    d["losses"] = safe_int(d["losses"]) + 1

        # Calculate win rates
        for group in analytics.values():
            for d in group.values():
                trades = safe_int(d.get("trades", 0))
                wins = safe_int(d.get("wins", 0))
                d["win_rate"] = safe_round(wins / trades * 100 if trades else 0, 1)

        self.history["analytics"] = analytics

    # ---------------- PUBLIC API ----------------
    def get_stats(self) -> Dict:
        """Get overall performance stats (guaranteed numeric types)."""
        stats = self.history.get("stats", {})
        return self._sanitize_stats(stats)

    def get_analytics(self) -> Dict:
        """Get detailed analytics (guaranteed structure)."""
        analytics = self.history.get("analytics", {})
        if not analytics:
            return self._empty_analytics()
        
        # Ensure structure exists
        analytics.setdefault("by_pair", {})
        analytics.setdefault("by_session", {})
        analytics.setdefault("by_confidence", {})
        
        return analytics

    def get_optimization_report(self) -> Dict:
        """Get advisory optimization recommendations."""
        try:
            return PerformanceOptimizer(self).get_optimal_parameters()
        except Exception as e:
            log.error(f"⚠️ Optimization failed: {e}")
            return PerformanceOptimizer(self)._default()

# ==========================================================
# LEGACY WRAPPER FOR TRADE_BEACON.PY
# ==========================================================
def track_performance(signals=None, risk_management=None) -> Dict:
    """
    Drop-in wrapper for trade_beacon.py.
    
    GUARANTEES:
    - Always returns {"stats": {...}, "analytics": {...}}
    - All numeric values are proper int/float types
    - Never raises type comparison errors
    - Handles all edge cases gracefully
    
    Args:
        signals: List of signal dicts to add
        risk_management: (unused, kept for API compatibility)
    
    Returns:
        Dict with 'stats' and 'analytics' keys
    """
    try:
        tracker = PerformanceTracker()
        
        # Add new signals if provided
        if signals:
            for sig in signals:
                try:
                    tracker.add_signal(sig)
                except Exception as e:
                    log.warning(f"⚠️ Failed to add signal: {e}")
            
            # Check all signals after adding
            try:
                tracker.check_signals()
            except Exception as e:
                log.warning(f"⚠️ Signal check failed: {e}")
        
        # Always return sanitized stats and analytics
        return {
            "stats": tracker.get_stats(),
            "analytics": tracker.get_analytics()
        }
        
    except Exception as e:
        log.error(f"⚠️ Performance tracking failed: {e}")
        # Return safe defaults on catastrophic failure
        return {
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pips": 0.0
            },
            "analytics": {
                "by_pair": {},
                "by_session": {},
                "by_confidence": {}
            }
        }

# ==========================================================
# STANDALONE EXECUTION
# ==========================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    tracker = PerformanceTracker()
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE TRACKING REPORT")
    print("="*80)
    
    # Display stats
    stats = tracker.get_stats()
    print(f"\n📈 Overall Stats:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
    print(f"  Win Rate: {stats['win_rate']}%")
    print(f"  Total Pips: {stats['total_pips']}")
    
    # Display analytics
    analytics = tracker.get_analytics()
    
    if analytics['by_pair']:
        print(f"\n🎯 Performance by Pair:")
        for pair, data in sorted(analytics['by_pair'].items(), 
                                key=lambda x: x[1]['win_rate'], 
                                reverse=True):
            print(f"  {pair}: {data['trades']} trades, "
                  f"{data['win_rate']}% win rate, "
                  f"{data['pips']} pips")
    
    # Get optimization report
    print(f"\n🔧 Optimization Report:")
    report = tracker.get_optimization_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "="*80)
    print("✅ Report complete")
    print("="*80 + "\n")
