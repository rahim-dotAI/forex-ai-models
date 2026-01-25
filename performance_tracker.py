"""
Performance Tracker v2.1.2 - Aligned with Trade Beacon v2.1.2
==============================================================

CHANGELOG:
- ✅ Version updated to 2.1.2
- ✅ Session normalization (LONDON→EUROPEAN, NEW_YORK→US)
- ✅ Confidence tier normalization (HIGH→VERY_STRONG, MEDIUM→MODERATE)
- ✅ EXPIRED signals tracking
- ✅ UTC-only datetime handling
- ✅ Deterministic ID validation
- ✅ Status transition guards
- ✅ Single expectancy metric
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd

log = logging.getLogger("performance-tracker")

TRACKER_VERSION = "2.1.2"

# Safe type conversion utilities
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

# Normalization functions for v2.1.2 alignment
def normalize_session(session: Optional[str]) -> str:
    """Normalize legacy session names to v2.1.2 taxonomy."""
    if not session:
        return "UNKNOWN"
    s = session.upper()
    if s in ("LONDON",):
        return "EUROPEAN"
    if s in ("NEW_YORK",):
        return "US"
    if s in ("ASIAN", "EUROPEAN", "OVERLAP", "US", "LATE_US"):
        return s
    return "UNKNOWN"

def normalize_confidence(conf: Optional[str]) -> str:
    """Normalize legacy confidence levels to v2.1.2 tiers."""
    if not conf:
        return "UNKNOWN"
    c = conf.upper()
    if c in ("HIGH", "VERY_HIGH"):
        return "VERY_STRONG"
    if c in ("MEDIUM",):
        return "MODERATE"
    if c in ("LOW", "WEAK"):
        return "WEAK"
    if c in ("VERY_STRONG", "STRONG", "MODERATE"):
        return c
    return "UNKNOWN"

class PerformanceTracker:
    """
    Signal-only performance tracker with trade resolution support.
    
    Aligned with Trade Beacon v2.1.2:
    - Session taxonomy: ASIAN, EUROPEAN, OVERLAP, US, LATE_US
    - Confidence tiers: VERY_STRONG, STRONG, MODERATE
    - Signal statuses: OPEN, EXPIRED, WIN, LOSS
    - UTC-only datetime handling
    - Deterministic SHA-1 signal IDs
    
    IMPORTANT:
    - No equity assumptions
    - No fake drawdowns
    - Stats only computed from RESOLVED signals (WIN/LOSS)
    - EXPIRED signals tracked separately
    - Designed for backtesting / paper simulation integration
    """

    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load()
        self._log_state()

    def _load(self) -> Dict:
        """Load history with migration support."""
        if not self.history_file.exists():
            return self._empty()

        try:
            with open(self.history_file) as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"⚠️ Could not load history: {e}, starting fresh")
            return self._empty()

        # Version migration
        old_version = data.get("version", "unknown")
        if old_version != TRACKER_VERSION:
            log.info(f"📦 Migrating from {old_version} to {TRACKER_VERSION}")
            data = self._migrate(data, old_version)
        
        data.setdefault("version", TRACKER_VERSION)
        data.setdefault("signals", [])
        data.setdefault("stats", self._empty_stats())
        data.setdefault("analytics", self._empty_analytics())
        data.setdefault("daily", {})
        data.setdefault("metadata", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        })

        return data

    def _migrate(self, data: Dict, from_version: str) -> Dict:
        """Migrate old data to v2.1.2 format."""
        log.info(f"🔄 Migrating {len(data.get('signals', []))} signals...")
        
        # Normalize all signals
        for signal in data.get("signals", []):
            # Normalize session
            if "session" in signal:
                signal["session"] = normalize_session(signal["session"])
            
            # Normalize confidence
            if "confidence" in signal:
                signal["confidence"] = normalize_confidence(signal["confidence"])
            
            # Ensure signal_id exists
            if "signal_id" not in signal and "id" in signal:
                signal["signal_id"] = signal["id"]
        
        data["version"] = TRACKER_VERSION
        log.info("✅ Migration complete")
        return data

    def _save(self):
        """Save history with UTC timestamp."""
        self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def _empty(self) -> Dict:
        """Create empty history structure."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": TRACKER_VERSION,
            "signals": [],
            "stats": self._empty_stats(),
            "analytics": self._empty_analytics(),
            "daily": {},
            "metadata": {"created_at": now, "last_updated": now}
        }

    def _empty_stats(self) -> Dict:
        """Empty stats structure."""
        return {
            "total_trades": 0,
            "total_resolved": 0,
            "wins": 0,
            "losses": 0,
            "expired": 0,
            "win_rate": 0.0,
            "total_pips": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy_pips": 0.0,
            "note": "Stats valid after statistical significance (100+ trades)"
        }

    def _empty_analytics(self) -> Dict:
        """Empty analytics structure."""
        return {
            "by_pair": {},
            "by_session": {},
            "by_confidence": {}
        }

    def register_signal(self, signal: Dict):
        """
        Register a signal safely (idempotent).
        Status must be: OPEN, EXPIRED, WIN, LOSS
        """
        signal_id = signal.get("id") or signal.get("signal_id")
        
        if not signal_id:
            raise ValueError("Signal must have deterministic id or signal_id")
        
        # Deterministic ID validation
        if len(signal_id) < 20:
            log.warning(f"⚠️ Non-deterministic signal ID detected: {signal_id}")
        
        # Check if already exists
        existing = self._find_signal(signal_id)
        if existing:
            log.debug(f"⏭️ Signal {signal_id} already registered")
            return

        # Normalize before storing
        if "session" in signal:
            signal["session"] = normalize_session(signal["session"])
        if "confidence" in signal:
            signal["confidence"] = normalize_confidence(signal["confidence"])
        
        # Add to signals list
        self.history["signals"].append(signal)

        # Track daily (UTC-only)
        today = datetime.now(timezone.utc).date().isoformat()
        self.history["daily"].setdefault(today, [])
        if signal_id not in self.history["daily"][today]:
            self.history["daily"][today].append(signal_id)

        self._recalculate()
        self._save()
        log.debug(f"✅ Signal {signal_id} registered")

    def record_trade(self, signal_id: str, pair: str, direction: str,
                     entry_price: float, exit_price: float, sl: float, tp: float,
                     outcome: str, pips: float, confidence: str = None,
                     score: int = None, session: str = None,
                     entry_time: str = None, exit_time: str = None, **kwargs):
        """
        Record trade outcome when it hits SL/TP or expires.
        
        Args:
            signal_id: Unique signal identifier (deterministic)
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
        # Find existing signal
        signal = self._find_signal(signal_id)
        
        # Status transition guard
        if signal and signal.get("status") in ("WIN", "LOSS", "EXPIRED"):
            log.warning(f"⚠️ Signal {signal_id} already resolved with status: {signal['status']}")
            return
        
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
                "confidence": normalize_confidence(confidence),
                "score": score,
                "session": normalize_session(session),
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

    def _recalculate(self):
        """
        Recalculate all statistics from resolved signals.
        
        RESOLVED = WIN or LOSS (used for stats)
        EXPIRED = tracked separately (not included in win rate)
        """
        resolved = [
            s for s in self.history["signals"]
            if s.get("status") in ("WIN", "LOSS")
        ]
        
        expired = [
            s for s in self.history["signals"]
            if s.get("status") == "EXPIRED"
        ]

        wins: List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics = self._empty_analytics()

        for s in resolved:
            pips = safe_float(s.get("pips"))
            total_pips += pips
            
            pair = s.get("pair", "UNKNOWN")
            session = normalize_session(s.get("session"))
            confidence = normalize_confidence(s.get("confidence"))

            if s["status"] == "WIN":
                wins.append(pips)
            else:
                losses.append(abs(pips))

            # Pair analytics
            analytics["by_pair"].setdefault(pair, {
                "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["by_pair"][pair]["trades"] += 1
            analytics["by_pair"][pair]["total_pips"] += pips
            if s["status"] == "WIN":
                analytics["by_pair"][pair]["wins"] += 1

            # Session analytics
            analytics["by_session"].setdefault(session, {
                "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["by_session"][session]["trades"] += 1
            analytics["by_session"][session]["total_pips"] += pips
            if s["status"] == "WIN":
                analytics["by_session"][session]["wins"] += 1

            # Confidence analytics
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
            "expired": len(expired),
            "win_rate": safe_round(win_rate, 1),
            "total_pips": safe_round(total_pips, 1),
            "avg_win": safe_round(avg_win, 1),
            "avg_loss": safe_round(avg_loss, 1),
            "expectancy_pips": safe_round(expectancy, 2),
            "validated": total >= 100
        }

        # Calculate win rates per dimension
        for group in analytics.values():
            for data in group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(
                        data["wins"] / data["trades"] * 100, 1
                    )

        self.history["analytics"] = analytics

    def export_to_csv(self, path="performance_export.csv") -> str:
        """Export all signals to CSV."""
        pd.DataFrame(self.history["signals"]).to_csv(path, index=False)
        log.info(f"📄 Exported to {path}")
        return path

    def get_dashboard_summary(self) -> Dict:
        """Get summary for dashboard integration."""
        return {
            "stats": self.history["stats"],
            "analytics": self.history["analytics"],
            "equity": {},  # Placeholder for compatibility
            "signals_total": len(self.history["signals"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "version": TRACKER_VERSION
        }

    def _log_state(self):
        """Log current tracker state."""
        open_sigs = [s for s in self.history["signals"] if s.get("status") == "OPEN"]
        resolved = [s for s in self.history["signals"] if s.get("status") in ("WIN", "LOSS")]
        expired = [s for s in self.history["signals"] if s.get("status") == "EXPIRED"]
        
        log.info(
            f"📊 Tracker v{TRACKER_VERSION} Loaded | "
            f"Total: {len(self.history['signals'])} | "
            f"Open: {len(open_sigs)} | "
            f"Resolved: {len(resolved)} | "
            f"Expired: {len(expired)}"
        )
