import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone, date
from functools import wraps
from typing import Dict, Any, Optional, List

import pandas as pd
import yfinance as yf

log = logging.getLogger("performance-tracker")

# ==========================================================
# SAFE TYPE CONVERSION UTILITIES
# ==========================================================
def safe_int(val: Any, default: int = 0) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return int(val)
    except Exception:
        return default

def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return float(val)
    except Exception:
        return default

def safe_round(val: Any, decimals: int = 2) -> float:
    return round(safe_float(val), decimals)

# ==========================================================
# FIXED: Pandas FutureWarning eliminated
# ==========================================================
def last(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    return None if pd.isna(val) else float(val)

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
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait = (2 ** attempt) * backoff_factor
                            log.warning(f"⚠️ Rate limit, retrying in {wait}s")
                            time.sleep(wait)
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

# ==========================================================
# YFINANCE CLEANING (CLAIMED → NOW REAL)
# ==========================================================
@retry_with_backoff()
def fetch_clean_price(symbol: str, period="1d", interval="5m") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    return df.dropna(subset=["High", "Low"])

# ==========================================================
# PERFORMANCE TRACKER
# ==========================================================
class PerformanceTracker:
    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.min_age_minutes = 10
        self.history = self._load()
        self._log_state()

    # ---------------- LOAD / SAVE ----------------
    def _load(self) -> Dict:
        if not self.history_file.exists():
            return self._empty()

        try:
            with open(self.history_file) as f:
                data = json.load(f)
        except Exception:
            return self._empty()

        data.setdefault("version", "2.0.5.1")
        data.setdefault("signals", [])
        data.setdefault("stats", self._empty_stats())
        data.setdefault("analytics", self._empty_analytics())
        data.setdefault("daily", {})
        data.setdefault("equity_curve", [])
        data.setdefault("metadata", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        })

        data["stats"] = self._sanitize_stats(data["stats"])
        return data

    def _save(self):
        self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.history["stats"] = self._sanitize_stats(self.history["stats"])
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    # ---------------- EMPTY STRUCTURES ----------------
    def _empty(self) -> Dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": "2.0.5.1",
            "signals": [],
            "stats": self._empty_stats(),
            "analytics": self._empty_analytics(),
            "daily": {},
            "equity_curve": [],
            "metadata": {"created_at": now, "last_updated": now}
        }

    def _empty_stats(self) -> Dict:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pips": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "expectancy_per_trade": 0.0
        }

    def _empty_analytics(self) -> Dict:
        return {
            "by_pair": {},
            "by_session": {},
            "by_confidence": {}
        }

    def _sanitize_stats(self, stats: Dict) -> Dict:
        return {
            "total_trades": safe_int(stats.get("total_trades")),
            "wins": safe_int(stats.get("wins")),
            "losses": safe_int(stats.get("losses")),
            "win_rate": safe_float(stats.get("win_rate")),
            "total_pips": safe_float(stats.get("total_pips")),
            "avg_win": safe_float(stats.get("avg_win")),
            "avg_loss": safe_float(stats.get("avg_loss")),
            "expectancy": safe_float(stats.get("expectancy")),
            "expectancy_per_trade": safe_float(stats.get("expectancy_per_trade", stats.get("expectancy")))
        }

    # ==========================================================
    # CORE UPDATE (IDEMPOTENT DAILY)
    # ==========================================================
    def update_closed_signal(self, signal: Dict):
        # ---- Direction validation (NO silent defaults)
        direction = signal.get("direction")
        if direction not in ("BUY", "SELL"):
            raise ValueError("Invalid direction")

        # ---- Confidence fallback chain
        confidence = (
            signal.get("confidence") or
            signal.get("confidence_score") or
            signal.get("score") or
            "UNKNOWN"
        )
        signal["confidence"] = confidence

        # ---- Daily idempotency
        today = str(date.today())
        self.history["daily"].setdefault(today, {"signal_ids": []})

        if signal["id"] in self.history["daily"][today]["signal_ids"]:
            log.info("⏭️ Signal already processed today")
            return

        # Save state for rollback
        original_signals = self.history["signals"].copy()
        original_daily = json.loads(json.dumps(self.history["daily"]))

        try:
            self.history["daily"][today]["signal_ids"].append(signal["id"])
            self.history["signals"].append(signal)

            self._recalculate_all()
            self._save()

        except Exception as e:
            log.error(f"❌ Failed to update signal: {e}")
            self.history["signals"] = original_signals
            self.history["daily"] = original_daily
            raise

    # ==========================================================
    # FULL RECALCULATION (SOURCE OF TRUTH)
    # ==========================================================
    def _recalculate_all(self):
        closed = [s for s in self.history["signals"] if s["status"] in ("WIN", "LOSS")]

        wins, losses = [], []
        equity = 0.0
        curve = []

        analytics = self._empty_analytics()

        for s in closed:
            pips = safe_float(s.get("pips"))
            equity += pips

            curve.append({
                "timestamp": s.get("closed_at"),
                "equity": safe_round(equity)
            })

            if s["status"] == "WIN":
                wins.append(pips)
            else:
                losses.append(abs(pips))

            # ---- Analytics
            pair = s.get("pair", "UNKNOWN")
            analytics["by_pair"].setdefault(pair, {"trades": 0, "wins": 0, "pips": 0.0})
            analytics["by_pair"][pair]["trades"] += 1
            analytics["by_pair"][pair]["pips"] += pips
            if s["status"] == "WIN":
                analytics["by_pair"][pair]["wins"] += 1

            # ✅ SESSION TRACKING ADDED
            session = s.get("session", "UNKNOWN")
            analytics["by_session"].setdefault(session, {"trades": 0, "wins": 0, "pips": 0.0})
            analytics["by_session"][session]["trades"] += 1
            analytics["by_session"][session]["pips"] += pips
            if s["status"] == "WIN":
                analytics["by_session"][session]["wins"] += 1

            conf = s.get("confidence", "UNKNOWN")
            analytics["by_confidence"].setdefault(conf, {"trades": 0, "wins": 0})
            analytics["by_confidence"][conf]["trades"] += 1
            if s["status"] == "WIN":
                analytics["by_confidence"][conf]["wins"] += 1

        total = len(closed)
        win_rate = (len(wins) / total * 100) if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)

        # --- Win rate calculation for analytics
        for pair_data in analytics["by_pair"].values():
            if pair_data["trades"] > 0:
                pair_data["win_rate"] = safe_round(pair_data["wins"] / pair_data["trades"] * 100)

        for session_data in analytics["by_session"].values():
            if session_data["trades"] > 0:
                session_data["win_rate"] = safe_round(session_data["wins"] / session_data["trades"] * 100)

        for conf_data in analytics["by_confidence"].values():
            if conf_data["trades"] > 0:
                conf_data["win_rate"] = safe_round(conf_data["wins"] / conf_data["trades"] * 100)

        self.history["stats"] = {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": safe_round(win_rate),
            "total_pips": safe_round(sum(wins) - sum(losses)),
            "avg_win": safe_round(avg_win),
            "avg_loss": safe_round(avg_loss),
            "expectancy": safe_round(expectancy),
            "expectancy_per_trade": safe_round(expectancy)
        }

        # --- Equity curve size limit (prevent unbounded growth)
        max_curve_points = 1000
        if len(curve) > max_curve_points:
            curve = curve[-max_curve_points:]

        self.history["analytics"] = analytics
        self.history["equity_curve"] = curve

    # ==========================================================
    # EXPORTS & DASHBOARD
    # ==========================================================
    def export_to_csv(self, path="performance_export.csv") -> str:
        closed = [s for s in self.history["signals"] if s["status"] in ("WIN", "LOSS")]
        pd.DataFrame(closed).to_csv(path, index=False)
        return path

    def get_dashboard_summary(self) -> Dict:
        stats = self.history["stats"]
        curve = self.history["equity_curve"]

        return {
            "stats": stats,
            "equity": {
                "current": curve[-1]["equity"] if curve else 0.0,
                "curve": curve[-50:]
            },
            "analytics": self.history["analytics"],
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    # ==========================================================
    # LOGGING
    # ==========================================================
    def _log_state(self):
        open_sigs = [s for s in self.history["signals"] if s.get("status") == "OPEN"]
        log.info(f"📊 Tracker Loaded | Total: {len(self.history['signals'])} | Open: {len(open_sigs)}")
