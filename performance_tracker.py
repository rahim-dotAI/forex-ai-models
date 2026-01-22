import json
import logging
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, List

import pandas as pd

log = logging.getLogger("performance-tracker")

# ==========================================================
# SAFE TYPE CONVERSION UTILITIES
# ==========================================================
def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default

def safe_round(val: Any, decimals: int = 2) -> float:
    return round(safe_float(val), decimals)

# ==========================================================
# PERFORMANCE TRACKER (WITH TRADE RESOLUTION)
# ==========================================================
class PerformanceTracker:
    """
    Signal-only performance tracker with trade resolution support.

    IMPORTANT:
    - No equity assumptions
    - No fake drawdowns
    - Stats only computed from RESOLVED signals
    - Designed for backtesting / paper simulation integration
    """

    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load()
        self._log_state()

    # ==========================================================
    # LOAD / SAVE
    # ==========================================================
    def _load(self) -> Dict:
        if not self.history_file.exists():
            return self._empty()

        try:
            with open(self.history_file) as f:
                data = json.load(f)
        except Exception:
            return self._empty()

        data.setdefault("version", "2.0.6")
        data.setdefault("signals", [])
        data.setdefault("stats", self._empty_stats())
        data.setdefault("analytics", self._empty_analytics())
        data.setdefault("daily", {})
        data.setdefault("metadata", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        })

        return data

    def _save(self):
        self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    # ==========================================================
    # EMPTY STRUCTURES
    # ==========================================================
    def _empty(self) -> Dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": "2.0.6",
            "signals": [],
            "stats": self._empty_stats(),
            "analytics": self._empty_analytics(),
            "daily": {},
            "metadata": {"created_at": now, "last_updated": now}
        }

    def _empty_stats(self) -> Dict:
        return {
            "total_trades": 0,
            "total_resolved": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pips": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "expectancy_per_trade": 0.0,
            "note": "Stats only valid after statistical validation"
        }

    def _empty_analytics(self) -> Dict:
        return {
            "by_pair": {},
            "by_session": {},
            "by_confidence": {}
        }

    # ==========================================================
    # SIGNAL INGEST (IDEMPOTENT)
    # ==========================================================
    def register_signal(self, signal: Dict):
        """
        Registers a signal safely.
        Status must be one of: OPEN, EXPIRED, WIN, LOSS
        """
        signal_id = signal.get("id") or signal.get("signal_id")
        
        if not signal_id:
            raise ValueError("Signal must have deterministic id or signal_id")

        # Check if already exists
        existing = self._find_signal(signal_id)
        if existing:
            log.debug(f"⏭️ Signal {signal_id} already registered")
            return

        # Add to signals list
        self.history["signals"].append(signal)

        # Track daily
        today = str(date.today())
        self.history["daily"].setdefault(today, [])
        if signal_id not in self.history["daily"][today]:
            self.history["daily"][today].append(signal_id)

        self._recalculate()
        self._save()
        log.debug(f"✅ Signal {signal_id} registered")

    # ==========================================================
    # ✅ NEW: TRADE RESOLUTION
    # ==========================================================
    def record_trade(self, signal_id: str, pair: str, direction: str,
                     entry_price: float, exit_price: float, sl: float, tp: float,
                     outcome: str, pips: float, confidence: str = None,
                     score: int = None, session: str = None,
                     entry_time: str = None, exit_time: str = None, **kwargs):
        """
        Records the outcome of a trade when it hits SL/TP or expires.
        
        Args:
            signal_id: Unique signal identifier
            pair: Currency pair
            direction: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            sl: Stop loss level
            tp: Take profit level
            outcome: WIN, LOSS, or EXPIRED
            pips: Pip profit/loss (negative for losses)
            confidence: Signal confidence level
            score: Signal score
            session: Market session
            entry_time: ISO timestamp of entry
            exit_time: ISO timestamp of exit
        """
        # Find existing signal or create new record
        signal = self._find_signal(signal_id)
        
        if signal:
            # Update existing signal
            signal["status"] = outcome
            signal["exit_price"] = exit_price
            signal["exit_time"] = exit_time or datetime.now(timezone.utc).isoformat()
            signal["pips"] = pips
            log.info(f"✅ Updated signal {signal_id}: {outcome} ({pips:+.1f} pips)")
        else:
            # Create new record if signal wasn't registered initially
            signal = {
                "id": signal_id,
                "signal_id": signal_id,
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "sl": sl,
                "tp": tp,
                "status": outcome,
                "pips": pips,
                "confidence": confidence,
                "score": score,
                "session": session,
                "timestamp": entry_time or datetime.now(timezone.utc).isoformat(),
                "exit_time": exit_time or datetime.now(timezone.utc).isoformat()
            }
            self.history["signals"].append(signal)
            log.info(f"✅ Recorded new trade {signal_id}: {outcome} ({pips:+.1f} pips)")
        
        self._recalculate()
        self._save()

    def _find_signal(self, signal_id: str) -> Optional[Dict]:
        """Find a signal by ID."""
        for signal in self.history["signals"]:
            if signal.get("id") == signal_id or signal.get("signal_id") == signal_id:
                return signal
        return None

    # ==========================================================
    # CORE RECALCULATION (SOURCE OF TRUTH)
    # ==========================================================
    def _recalculate(self):
        """Recalculate all statistics from resolved signals."""
        resolved = [
            s for s in self.history["signals"]
            if s.get("status") in ("WIN", "LOSS")
        ]

        wins: List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics = self._empty_analytics()

        for s in resolved:
            pips = safe_float(s.get("pips"))
            total_pips += pips
            
            pair = s.get("pair", "UNKNOWN")
            session = s.get("session", "UNKNOWN")
            confidence = s.get("confidence", "UNKNOWN")

            if s["status"] == "WIN":
                wins.append(pips)
            else:
                losses.append(abs(pips))

            # ---- Pair analytics
            analytics["by_pair"].setdefault(pair, {
                "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["by_pair"][pair]["trades"] += 1
            analytics["by_pair"][pair]["total_pips"] += pips
            if s["status"] == "WIN":
                analytics["by_pair"][pair]["wins"] += 1

            # ---- Session analytics
            analytics["by_session"].setdefault(session, {
                "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["by_session"][session]["trades"] += 1
            analytics["by_session"][session]["total_pips"] += pips
            if s["status"] == "WIN":
                analytics["by_session"][session]["wins"] += 1

            # ---- Confidence analytics
            analytics["by_confidence"].setdefault(confidence, {
                "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["by_confidence"][confidence]["trades"] += 1
            analytics["by_confidence"][confidence]["total_pips"] += pips
            if s["status"] == "WIN":
                analytics["by_confidence"][confidence]["wins"] += 1

        total = len(resolved)
        win_rate = (len(wins) / total * 100) if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = (
            (win_rate / 100) * avg_win -
            (1 - win_rate / 100) * avg_loss
        ) if total else 0.0

        self.history["stats"] = {
            "total_trades": total,
            "total_resolved": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": safe_round(win_rate, 1),
            "total_pips": safe_round(total_pips, 1),
            "avg_win": safe_round(avg_win, 1),
            "avg_loss": safe_round(avg_loss, 1),
            "expectancy": safe_round(expectancy, 2),
            "expectancy_per_trade": safe_round(expectancy, 2),
            "validated": total >= 100
        }

        # ---- Win rates per dimension
        for group in analytics.values():
            for data in group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(
                        data["wins"] / data["trades"] * 100, 1
                    )

        self.history["analytics"] = analytics

    # ==========================================================
    # EXPORTS
    # ==========================================================
    def export_to_csv(self, path="performance_export.csv") -> str:
        pd.DataFrame(self.history["signals"]).to_csv(path, index=False)
        return path

    def get_dashboard_summary(self) -> Dict:
        return {
            "stats": self.history["stats"],
            "analytics": self.history["analytics"],
            "equity": {},  # Placeholder for compatibility
            "signals_total": len(self.history["signals"]),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    # ==========================================================
    # LOGGING
    # ==========================================================
    def _log_state(self):
        open_sigs = [s for s in self.history["signals"] if s.get("status") == "OPEN"]
        resolved = [s for s in self.history["signals"] if s.get("status") in ("WIN", "LOSS")]
        log.info(
            f"📊 Tracker Loaded | Total: {len(self.history['signals'])} | "
            f"Open: {len(open_sigs)} | Resolved: {len(resolved)}"
        )
