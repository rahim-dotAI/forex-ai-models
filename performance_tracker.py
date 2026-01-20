"""
Performance Tracking System for Trade Beacon v2.0.4-FINAL

Tracks signal outcomes, calculates win rate, pips, drawdown,
and produces advisory optimization insights.

ALIGNMENTS (v2.0.4):
- Version consistency with engine
- Signal age aligned to 10 minutes (600s)
- Numeric confidence → labeled buckets
- Advisory-only optimization (no auto mutation)
- All v2.0.3 functionality retained
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from functools import wraps

import pandas as pd
import yfinance as yf

log = logging.getLogger("performance-tracker")

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
# HELPERS
# ==========================================================
def ensure_series(data):
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data.squeeze()

# ==========================================================
# PERFORMANCE OPTIMIZER (ADVISORY)
# ==========================================================
class PerformanceOptimizer:
    def __init__(self, tracker):
        self.tracker = tracker
        self.min_trades = 30

    def get_optimal_parameters(self) -> Dict:
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

    def _best_sessions(self, data):
        ranked = []
        for s, d in data.items():
            if d["trades"] >= 5:
                ranked.append((s, d["win_rate"]))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:3]]

    def _best_pairs(self, data):
        ranked = []
        for p, d in data.items():
            if d["trades"] >= 5:
                ranked.append((p, d["win_rate"], d["pips"]))
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [p for p, _, _ in ranked if _ >= 55]

    def _best_confidence(self, data):
        for level in ["EXCELLENT", "STRONG", "GOOD"]:
            stats = data.get(level)
            if stats and stats["trades"] >= 5 and stats["win_rate"] >= 55:
                return level
        return "GOOD"

    def _risk_reward(self, signals):
        buckets = {"<1.5": [], "1.5-2.0": [], "2.0+": []}
        for s in signals:
            rr = s.get("risk_reward", 0)
            win = s["status"] == "WIN"
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
            summary[k] = {"trades": len(v), "win_rate": round(wr, 1)}
            if wr > best_wr and len(v) >= 5:
                best_wr = wr
                best = k

        return {
            "by_range": summary,
            "recommended_range": best or "1.5-2.0",
            "recommended_min_rr": 1.5 if best != "<1.5" else 2.0
        }

    def _threshold_advice(self, signals):
        wins = sum(1 for s in signals if s["status"] == "WIN")
        total = len(signals)
        wr = wins / total * 100 if total else 0

        if wr < 50:
            return {"action": "RAISE", "amount": 5, "win_rate": round(wr, 1)}
        if wr > 65:
            return {"action": "LOWER", "amount": 3, "win_rate": round(wr, 1)}
        return {"action": "MAINTAIN", "amount": 0, "win_rate": round(wr, 1)}

    def _default(self):
        return {
            "advisory_only": True,
            "recommended_sessions": [],
            "optimal_pairs": [],
            "min_confidence": "GOOD",
            "risk_reward_insights": {},
            "threshold_adjustment": {"action": "MAINTAIN", "amount": 0},
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
        self.min_age_minutes = 10  # v2.0.4 alignment
        self.history = self._load()

    # ---------------- LOAD / SAVE ----------------
    def _load(self):
        if not self.history_file.exists():
            return self._empty()

        try:
            with open(self.history_file) as f:
                data = json.load(f)
            data["version"] = "2.0.4"
            return data
        except Exception:
            return self._empty()

    def _empty(self):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": "2.0.4",
            "signals": [],
            "stats": {},
            "daily": {},
            "analytics": {"by_pair": {}, "by_session": {}, "by_confidence": {}},
            "metadata": {"created_at": now, "last_updated": now}
        }

    def _save(self):
        self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    # ---------------- CONFIDENCE NORMALIZATION ----------------
    @staticmethod
    def _confidence_bucket(score: float) -> str:
        if score >= 85:
            return "EXCELLENT"
        if score >= 70:
            return "STRONG"
        if score >= 55:
            return "GOOD"
        return "MODERATE"

    # ---------------- SIGNAL INGEST ----------------
    def add_signal(self, signal: Dict):
        sid = signal.get("signal_id") or f"{signal['pair']}_{signal['timestamp']}"
        if any(s["id"] == sid for s in self.history["signals"]):
            return

        self.history["signals"].append({
            "id": sid,
            "pair": signal["pair"],
            "direction": signal["direction"],
            "entry_price": signal["entry_price"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "risk_reward": signal.get("risk_reward", 0),
            "confidence_score": signal.get("confidence", 0),
            "confidence": self._confidence_bucket(signal.get("confidence", 0)),
            "session": signal.get("session"),
            "timestamp": signal["timestamp"],
            "status": "OPEN",
            "pips": 0.0,
            "closed_at": None,
            "exit_index": None
        })
        self._save()

    # ---------------- PRICE CHECK ----------------
    def check_signals(self):
        open_signals = [s for s in self.history["signals"] if s["status"] == "OPEN"]
        for s in open_signals:
            self._check_one(s)
        self._stats()
        self._analytics()
        self._save()

    @retry_with_backoff()
    def _download(self, pair, start):
        return yf.download(pair + "=X", start=start, interval="1m",
                           progress=False, auto_adjust=True, threads=False)

    def _check_one(self, s):
        ts = datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age < self.min_age_minutes:
            return

        df = self._download(s["pair"], ts)
        if df.empty:
            return

        entry, sl, tp = s["entry_price"], s["sl"], s["tp"]
        pip = 0.01 if "JPY" in s["pair"] else 0.0001

        for idx, r in df.iterrows():
            hi, lo = r["High"], r["Low"]
            if s["direction"] == "BUY":
                if lo <= sl:
                    self._close(s, "LOSS", sl, (sl - entry) / pip, idx)
                    return
                if hi >= tp:
                    self._close(s, "WIN", tp, (tp - entry) / pip, idx)
                    return
            else:
                if hi >= sl:
                    self._close(s, "LOSS", sl, (entry - sl) / pip, idx)
                    return
                if lo <= tp:
                    self._close(s, "WIN", tp, (entry - tp) / pip, idx)
                    return

    def _close(self, s, status, price, pips, idx):
        s["status"] = status
        s["closed_at"] = datetime.now(timezone.utc).isoformat()
        s["pips"] = round(pips, 1)
        s["exit_index"] = str(idx)

        day = pd.Timestamp(idx).date().isoformat()
        self.history["daily"].setdefault(day, {"pips": 0, "trades": 0})
        self.history["daily"][day]["pips"] += s["pips"]
        self.history["daily"][day]["trades"] += 1

    # ---------------- STATS / ANALYTICS ----------------
    def _stats(self):
        closed = [s for s in self.history["signals"] if s["status"] in ("WIN", "LOSS")]
        wins = sum(1 for s in closed if s["status"] == "WIN")
        pips = sum(s["pips"] for s in closed)
        total = len(closed)
        self.history["stats"] = {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "total_pips": round(pips, 1)
        }

    def _analytics(self):
        analytics = {"by_pair": {}, "by_session": {}, "by_confidence": {}}
        for s in self.history["signals"]:
            if s["status"] not in ("WIN", "LOSS"):
                continue

            for key, val in [
                ("by_pair", s["pair"]),
                ("by_session", s.get("session", "UNKNOWN")),
                ("by_confidence", s["confidence"])
            ]:
                analytics[key].setdefault(val, {"wins": 0, "losses": 0, "pips": 0, "trades": 0})
                d = analytics[key][val]
                d["trades"] += 1
                d["pips"] += s["pips"]
                d["wins" if s["status"] == "WIN" else "losses"] += 1

        for group in analytics.values():
            for d in group.values():
                d["win_rate"] = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0

        self.history["analytics"] = analytics

    # ---------------- PUBLIC ----------------
    def get_stats(self):
        return self.history.get("stats", {})

    def get_analytics(self):
        return self.history.get("analytics", {})

    def get_optimization_report(self):
        return PerformanceOptimizer(self).get_optimal_parameters()


# ==========================================================
# STANDALONE
# ==========================================================
if __name__ == "__main__":
    tracker = PerformanceTracker()
    report = tracker.get_optimization_report()
    print(json.dumps(report, indent=2))
