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
# PERFORMANCE TRACKER (SIGNAL-ONLY SAFE)
# ==========================================================
class PerformanceTracker:
    """
    Signal-only performance tracker.

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
            "total_resolved": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
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
        Status must be one of:
        OPEN, EXPIRED, WIN, LOSS
        """

        if "id" not in signal:
            raise ValueError("Signal must have deterministic id")

        today = str(date.today())
        self.history["daily"].setdefault(today, [])

        if signal["id"] in self.history["daily"][today]:
            log.info("⏭️ Signal already registered today")
            return

        self.history["daily"][today].append(signal["id"])
        self.history["signals"].append(signal)

        self._recalculate()
        self._save()

    # ==========================================================
    # CORE RECALCULATION (SOURCE OF TRUTH)
    # ==========================================================
    def _recalculate(self):
        resolved = [
            s for s in self.history["signals"]
            if s.get("status") in ("WIN", "LOSS")
        ]

        wins: List[float] = []
        losses: List[float] = []
        analytics = self._empty_analytics()

        for s in resolved:
            pips = safe_float(s.get("pips"))
            pair = s.get("pair", "UNKNOWN")
            session = s.get("session", "UNKNOWN")
            confidence = s.get("confidence", "UNKNOWN")

            if s["status"] == "WIN":
                wins.append(pips)
            else:
                losses.append(abs(pips))

            # ---- Pair analytics
            analytics["by_pair"].setdefault(pair, {"trades": 0, "wins": 0})
            analytics["by_pair"][pair]["trades"] += 1
            if s["status"] == "WIN":
                analytics["by_pair"][pair]["wins"] += 1

            # ---- Session analytics
            analytics["by_session"].setdefault(session, {"trades": 0, "wins": 0})
            analytics["by_session"][session]["trades"] += 1
            if s["status"] == "WIN":
                analytics["by_session"][session]["wins"] += 1

            # ---- Confidence analytics
            analytics["by_confidence"].setdefault(confidence, {"trades": 0, "wins": 0})
            analytics["by_confidence"][confidence]["trades"] += 1
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
            "total_resolved": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": safe_round(win_rate),
            "avg_win_pips": safe_round(avg_win),
            "avg_loss_pips": safe_round(avg_loss),
            "expectancy_per_trade": safe_round(expectancy),
            "validated": total >= 100
        }

        # ---- Win rates per dimension
        for group in analytics.values():
            for data in group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(
                        data["wins"] / data["trades"] * 100
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
