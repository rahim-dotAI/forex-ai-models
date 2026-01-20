"""
Performance Tracking System for Trade Beacon v2.0.3
Tracks signal outcomes, calculates win rate, pips, and performance stats

UPDATES (v2.0.3):
- Exact daily pips attribution using candle timestamp
- exit_index tracks which candle closed the signal
- Minor cleanup for High/Low access
- All v2.0.2 fixes retained
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from functools import wraps
import time
import pandas as pd
import yfinance as yf

log = logging.getLogger("performance-tracker")

# =========================
# RETRY DECORATOR
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
                        raise
            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator

# =========================
# DATA SHAPE HELPER
# =========================
def ensure_series(data):
    """Convert yfinance data to 1D Series"""
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data.squeeze()

# =========================
# PERFORMANCE TRACKER
# =========================
class PerformanceTracker:
    def __init__(self, history_file: str = "signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load_history()
        self.min_age_minutes = 15  # Match signal timeframe

    def _load_history(self) -> Dict:
        if not self.history_file.exists():
            return self._empty_history()
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            if not isinstance(data.get("signals"), list):
                raise ValueError("Invalid history structure")
            version = data.get("version", "1.0.0")
            if version not in ["2.0.0", "2.0.2", "2.0.3"]:
                log.warning(f"⚠️ History version {version} - migrating to 2.0.3")
            return data
        except Exception as e:
            log.error(f"Failed to load history: {e}")
            if self.history_file.exists():
                backup = self.history_file.with_suffix('.json.bak')
                self.history_file.rename(backup)
                log.warning(f"Backed up corrupted file to {backup}")
            return self._empty_history()

    def _empty_history(self) -> Dict:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "version": "2.0.3",
            "signals": [],
            "stats": {"total_trades":0,"winning_trades":0,"losing_trades":0,"total_pips":0.0,"win_rate":0.0},
            "daily": {},
            "analytics": {"by_pair":{},"by_session":{},"by_confidence":{}},
            "metadata": {"created_at": now_iso, "last_updated": now_iso}
        }

    def _save_history(self):
        try:
            self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save history: {e}")

    def add_signal(self, signal: Dict):
        signal_id = signal.get('signal_id') or f"{signal['pair']}_{signal['timestamp']}"
        existing = next((s for s in self.history["signals"] if s.get("id") == signal_id), None)
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
            "score": signal.get("score", 0),
            "confidence": signal.get("confidence", 0),
            "timestamp": signal["timestamp"],
            "status": "OPEN",
            "outcome": None,
            "exit_reason": None,
            "pips": 0.0,
            "closed_at": None,
            "closed_price": None,
            "risk_reward": signal.get("risk_reward", 0.0),
            "hold_time": signal.get("hold_time"),
            "eligible_modes": signal.get("eligible_modes", []),
            "session": signal.get("session"),
            "atr": signal.get("atr", 0.0),
            "spread": signal.get("spread", 0.0),
            "exit_index": None  # ✅ Tracks candle closing the signal
        }

        self.history["signals"].append(tracked_signal)
        log.info(f"✅ Added signal: {signal['pair']} {signal['direction']} (ID: {signal_id})")
        self._save_history()

    # =========================
    # PRICE CHECK / SIGNAL OUTCOME
    # =========================
    def check_signals(self):
        open_signals = [s for s in self.history["signals"] if s["status"] == "OPEN"]
        if not open_signals:
            log.info("No open signals to check")
            return
        log.info(f"🔍 Checking {len(open_signals)} open signals...")
        for signal in open_signals:
            self._check_signal_outcome(signal)
        self._calculate_stats()
        self._calculate_analytics()
        self._save_history()

    @retry_with_backoff()
    def _download_price_data(self, pair_symbol: str, start_time: datetime):
        df = yf.download(
            pair_symbol,
            start=start_time,
            interval="1m",
            progress=False,
            auto_adjust=True,
            threads=False
        )
        return df

    def _check_signal_outcome(self, signal: Dict):
        pair_symbol = signal["pair"] + "=X"
        try:
            signal_time = datetime.fromisoformat(signal["timestamp"].replace("Z", "+00:00"))
            age_minutes = (datetime.now(timezone.utc) - signal_time).total_seconds() / 60
            if age_minutes < self.min_age_minutes:
                log.debug(f"⏳ {signal['pair']} too young ({age_minutes:.1f}m), skipping")
                return

            df = self._download_price_data(pair_symbol, signal_time)
            if df.empty:
                log.warning(f"⚠️ No data for {signal['pair']}")
                return
            df = df[df.index >= signal_time]
            if len(df) == 0: return

            close = ensure_series(df["Close"])
            high = ensure_series(df["High"])
            low = ensure_series(df["Low"])
            if len(close) == 0 or len(high) == 0 or len(low) == 0:
                log.warning(f"⚠️ Empty price data for {signal['pair']}")
                return

            direction = signal["direction"]
            entry = signal["entry_price"]
            tp = signal["tp"]
            sl = signal["sl"]

            for idx, row in df.iterrows():
                candle_high = float(row["High"])
                candle_low = float(row["Low"])

                if direction == "BUY":
                    if candle_low <= sl:
                        pips = self._calculate_pips(signal['pair'], entry, sl, direction)
                        signal["exit_index"] = idx
                        self._close_signal(signal, "LOSS", sl, pips, "SL")
                        log.info(f"❌ LOSS: {signal['pair']} BUY - SL hit at {sl} ({pips:.1f} pips)")
                        return
                    if candle_high >= tp:
                        pips = self._calculate_pips(signal['pair'], entry, tp, direction)
                        signal["exit_index"] = idx
                        self._close_signal(signal, "WIN", tp, pips, "TP")
                        log.info(f"✅ WIN: {signal['pair']} BUY - TP hit at {tp} (+{pips:.1f} pips)")
                        return
                else:
                    if candle_high >= sl:
                        pips = self._calculate_pips(signal['pair'], entry, sl, direction)
                        signal["exit_index"] = idx
                        self._close_signal(signal, "LOSS", sl, pips, "SL")
                        log.info(f"❌ LOSS: {signal['pair']} SELL - SL hit at {sl} ({pips:.1f} pips)")
                        return
                    if candle_low <= tp:
                        pips = self._calculate_pips(signal['pair'], entry, tp, direction)
                        signal["exit_index"] = idx
                        self._close_signal(signal, "WIN", tp, pips, "TP")
                        log.info(f"✅ WIN: {signal['pair']} SELL - TP hit at {tp} (+{pips:.1f} pips)")
                        return

            current_price = float(close.iloc[-1])
            log.debug(f"📊 {signal['pair']} {direction} still open (current: {current_price:.5f})")
            if age_minutes > (7 * 24 * 60):
                signal["exit_index"] = df.index[-1]
                self._close_signal(signal, "EXPIRED", current_price, 0.0, "EXPIRED")
                log.info(f"⏰ EXPIRED: {signal['pair']} {direction} ({age_minutes/60/24:.1f} days)")

        except Exception as e:
            log.error(f"Error checking {signal['pair']}: {e}")

    def _close_signal(self, signal: Dict, outcome: str, close_price: float, pips: float, exit_reason: str):
        signal["status"] = outcome
        signal["outcome"] = outcome
        signal["exit_reason"] = exit_reason
        signal["closed_at"] = datetime.now(timezone.utc).isoformat()
        signal["closed_price"] = close_price
        signal["pips"] = pips

        # ✅ Use candle timestamp if available for daily stats
        if signal.get("exit_index") is not None:
            close_date = pd.Timestamp(signal["exit_index"]).date().isoformat()
        else:
            close_date = datetime.now(timezone.utc).date().isoformat()

        if close_date not in self.history["daily"]:
            self.history["daily"][close_date] = {"pips": 0.0, "trades": 0, "wins": 0, "losses": 0}

        if outcome == "WIN":
            self.history["daily"][close_date]["pips"] += pips
            self.history["daily"][close_date]["trades"] += 1
            self.history["daily"][close_date]["wins"] += 1
        elif outcome == "LOSS":
            self.history["daily"][close_date]["pips"] += pips
            self.history["daily"][close_date]["trades"] += 1
            self.history["daily"][close_date]["losses"] += 1

    # =========================
    # PIP CALCULATION
    # =========================
    def _calculate_pips(self, pair: str, entry: float, exit: float, direction: str) -> float:
        if entry <= 0 or exit <= 0:
            log.error(f"Invalid prices: entry={entry}, exit={exit}")
            return 0.0
        pip_value = 0.01 if "JPY" in pair else 0.0001
        diff = exit - entry
        if direction == "SELL":
            diff = -diff
        pips = diff / pip_value
        if abs(pips) > 1000:
            log.warning(f"⚠️ Suspicious pip value: {pips:.1f} for {pair}")
        return round(pips, 1)

    # =========================
    # STATS & ANALYTICS
    # =========================
    def _calculate_stats(self):
        closed_signals = [s for s in self.history["signals"] if s["status"] in ["WIN", "LOSS"]]
        if not closed_signals:
            self.history["stats"] = {"total_trades":0,"winning_trades":0,"losing_trades":0,"total_pips":0.0,"win_rate":0.0}
            return
        winning = [s for s in closed_signals if s["status"] == "WIN"]
        losing = [s for s in closed_signals if s["status"] == "LOSS"]
        total_pips = sum(s["pips"] for s in closed_signals)
        win_rate = (len(winning) / len(closed_signals) * 100)
        self.history["stats"] = {
            "total_trades": len(closed_signals),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_pips": round(total_pips, 1),
            "win_rate": round(win_rate, 1)
        }
        log.info(f"📊 Stats: {len(closed_signals)} trades | Win Rate: {win_rate:.1f}% | Total Pips: {total_pips:.1f}")

    def _calculate_analytics(self):
        closed_signals = [s for s in self.history["signals"] if s["status"] in ["WIN","LOSS"]]
        if not closed_signals: return
        analytics = {"by_pair": {}, "by_session": {}, "by_confidence": {}}

        for signal in closed_signals:
            # by pair
            pair = signal["pair"]
            analytics["by_pair"].setdefault(pair, {"wins":0,"losses":0,"pips":0.0,"trades":0})
            analytics["by_pair"][pair]["trades"] += 1
            analytics["by_pair"][pair]["pips"] += signal.get("pips",0)
            if signal["status"] == "WIN": analytics["by_pair"][pair]["wins"] +=1
            else: analytics["by_pair"][pair]["losses"] +=1
            # by session
            session = signal.get("session","UNKNOWN")
            analytics["by_session"].setdefault(session, {"wins":0,"losses":0,"pips":0.0,"trades":0})
            analytics["by_session"][session]["trades"] +=1
            analytics["by_session"][session]["pips"] += signal.get("pips",0)
            if signal["status"]=="WIN": analytics["by_session"][session]["wins"]+=1
            else: analytics["by_session"][session]["losses"]+=1
            # by confidence
            confidence = signal.get("confidence","UNKNOWN")
            analytics["by_confidence"].setdefault(confidence, {"wins":0,"losses":0,"pips":0.0,"trades":0})
            analytics["by_confidence"][confidence]["trades"]+=1
            analytics["by_confidence"][confidence]["pips"]+=signal.get("pips",0)
            if signal["status"]=="WIN": analytics["by_confidence"][confidence]["wins"]+=1
            else: analytics["by_confidence"][confidence]["losses"]+=1

        # calculate win rates
        for category in analytics.values():
            for k,d in category.items():
                total=d["trades"]
                d["win_rate"]=round((d["wins"]/total*100 if total>0 else 0),1)

        self.history["analytics"]=analytics

    # =========================
    # PUBLIC GETTERS
    # =========================
    def get_stats(self) -> Dict:
        return self.history.get("stats", {"total_trades":0,"winning_trades":0,"losing_trades":0,"total_pips":0.0,"win_rate":0.0})

    def get_analytics(self) -> Dict:
        return self.history.get("analytics", {"by_pair":{},"by_session":{},"by_confidence":{}})

    def get_daily_pips(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        return self.history.get("daily",{}).get(today,{}).get("pips",0.0)

    def get_open_signals(self) -> List[Dict]:
        return [s for s in self.history["signals"] if s["status"]=="OPEN"]

    def get_risk_metrics(self, current_signals: List[Dict]) -> Dict:
        if not current_signals: return {"total_risk_pips":0.0,"max_drawdown":0.0,"average_risk_reward":0.0}
        total_risk_pips=0.0
        risk_rewards=[]
        for s in current_signals:
            entry=s.get("entry_price",0)
            sl=s.get("sl",0)
            pair=s.get("pair","")
            dir=s.get("direction","BUY")
            if entry>0 and sl>0:
                total_risk_pips+=abs(self._calculate_pips(pair,entry,sl,dir))
            rr=s.get("risk_reward",0)
            if rr>0: risk_rewards.append(rr)
        avg_rr=sum(risk_rewards)/len(risk_rewards) if risk_rewards else 0.0
        max_drawdown=self._calculate_max_drawdown()
        return {"total_risk_pips":round(total_risk_pips,1),"max_drawdown":round(max_drawdown,1),"average_risk_reward":round(avg_rr,2)}

    def _calculate_max_drawdown(self) -> float:
        closed_signals=[s for s in self.history["signals"] if s["status"] in ["WIN","LOSS"]]
        if not closed_signals: return 0.0
        sorted_signals=sorted(closed_signals,key=lambda x:x.get("closed_at",x.get("timestamp","")))
        cum_pips=0
        peak=0
        max_dd=0
        for s in sorted_signals:
            cum_pips+=s.get("pips",0)
            peak=max(peak,cum_pips)
            max_dd=max(max_dd,peak-cum_pips)
        return max_dd

    def cleanup_old_signals(self, days: int = 30):
        cutoff=datetime.now(timezone.utc)-timedelta(days=days)
        before_count=len(self.history["signals"])
        self.history["signals"]=[s for s in self.history["signals"] if datetime.fromisoformat(s["timestamp"].replace("Z","+00:00"))>cutoff]
        removed=before_count-len(self.history["signals"])
        if removed>0:
            log.info(f"🧹 Cleaned up {removed} old signals (>{days} days)")
            self._save_history()

    def reset_stats(self, backup: bool = True):
        if backup and self.history_file.exists():
            backup_file=self.history_file.with_name(f"signal_history_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
            with open(self.history_file,'r') as f: data=f.read()
            with open(backup_file,'w') as f: f.write(data)
            log.info(f"📦 Backed up old stats to {backup_file}")
        self.history=self._empty_history()
        self._save_history()
        log.info("🔄 Performance stats reset")

# =========================
# STANDALONE TRACKER FUNCTION
# =========================
def track_performance(signals: List[Dict]) -> Dict:
    tracker=PerformanceTracker()
    for s in signals: tracker.add_signal(s)
    tracker.check_signals()
    tracker.cleanup_old_signals(days=30)
    stats=tracker.get_stats()
    daily_pips=tracker.get_daily_pips()
    open_signals=tracker.get_open_signals()
    risk_metrics=tracker.get_risk_metrics(open_signals)
    return {
        "stats":{
            "total_trades": stats.get("total_trades",0),
            "win_rate": stats.get("win_rate",0.0),
            "total_pips": stats.get("total_pips",0.0),
            "wins": stats.get("winning_trades",0),
            "losses": stats.get("losing_trades",0)
        },
        "risk_management":{
            "daily_pips": round(daily_pips,1),
            "total_risk_pips": risk_metrics["total_risk_pips"],
            "max_drawdown": risk_metrics["max_drawdown"],
            "average_risk_reward": risk_metrics["average_risk_reward"]
        }
    }
