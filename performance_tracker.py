"""
Performance Tracker v2.3.0-MTF-GEMINI - Aligned with Trade Beacon v2.3.0-MTF-GEMINI
============================================================================

CHANGELOG v2.3.0-MTF-GEMINI (based on v2.2.2-SAFE):
- Version bumped to 2.3.0-MTF-GEMINI
- New signal fields added to schema:
    mtf_details:      dict of per-timeframe trend values (1h/30m/4h/1d/1wk)
    gemini_usd_bias:  BULLISH/BEARISH/NEUTRAL/None
    gemini_engine:    "gemini-2.5-flash" / "finbert-fallback" / None
- New analytics group: by_mtf_confirmed (WR when all TFs confirmed vs partial)
- New analytics group: by_gemini_bias (WR when gemini said BULLISH vs BEARISH)
- AUTOMATIC MIGRATION: migrates v2.2.2-SAFE signals to v2.3.0-MTF-GEMINI on load
  New fields default to None/False for historical signals
- All v2.2.2-SAFE fixes retained:
    avg_win_pips / avg_loss_pips aliases
    expectancy alias alongside expectancy_pips
    Tier thresholds: A+:80, A:72, B:55
    resolve dedup guard
    signal resolution uses high/low window
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd

log = logging.getLogger("performance-tracker")

TRACKER_VERSION = "2.3.0-MTF-GEMINI"

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

def normalize_session(session: Optional[str]) -> str:
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

def normalize_tier(tier: Optional[str]) -> str:
    if not tier:
        return "C"
    t = tier.upper()
    if t in ("A+", "A", "B", "C"):
        return t
    return "C"

def map_confidence_to_tier(confidence: str) -> str:
    mapping = {
        'VERY_STRONG': 'A+',
        'STRONG': 'A',
        'MODERATE': 'B',
        'WEAK': 'C',
        'UNKNOWN': 'C'
    }
    return mapping.get(confidence, 'C')

def map_score_to_tier(score: int) -> str:
    """Map score to tier — v2.3.0 thresholds: A+:80, A:72, B:55, C:<55"""
    if score >= 80:
        return "A+"
    elif score >= 72:
        return "A"
    elif score >= 55:
        return "B"
    else:
        return "C"

def map_score_to_modes(score: int) -> List[str]:
    if score >= 55:
        return ['aggressive', 'conservative']
    else:
        return []


class PerformanceTracker:
    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load()
        self._log_state()

    def _load(self) -> Dict:
        if not self.history_file.exists():
            return self._empty()
        try:
            with open(self.history_file) as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"Could not load history: {e}, starting fresh")
            return self._empty()
        old_version = data.get("version", "unknown")
        if old_version != TRACKER_VERSION:
            log.info(f"Migrating from {old_version} to {TRACKER_VERSION}")
            data = self._migrate(data, old_version)
            self.history_file.parent.mkdir(exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)
            log.info(f"Migration saved to {self.history_file}")
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
        log.info(f"Migrating {len(data.get('signals', []))} signals from {from_version}...")
        migrated_count = 0
        for signal in data.get("signals", []):
            needs_migration = False
            if "session" in signal:
                old_session = signal["session"]
                signal["session"] = normalize_session(signal["session"])
                if old_session != signal["session"]:
                    needs_migration = True
            if "confidence" in signal:
                old_conf = signal["confidence"]
                signal["confidence"] = normalize_confidence(signal["confidence"])
                if old_conf != signal["confidence"]:
                    needs_migration = True
            if "score" in signal:
                score = safe_int(signal["score"])
                new_tier = map_score_to_tier(score)
                if signal.get("tier") != new_tier:
                    signal["tier"] = new_tier
                    needs_migration = True
            elif "tier" not in signal or not signal["tier"]:
                conf = normalize_confidence(signal.get("confidence", "MODERATE"))
                signal["tier"] = map_confidence_to_tier(conf)
                needs_migration = True
            if "eligible_modes" not in signal or not signal["eligible_modes"]:
                score = safe_int(signal.get("score", 0))
                signal["eligible_modes"] = map_score_to_modes(score)
                needs_migration = True
            if "signal_id" not in signal and "id" in signal:
                signal["signal_id"] = signal["id"]
                needs_migration = True

            # v2.2.2-SAFE fields
            signal.setdefault("sentiment_applied",     False)
            signal.setdefault("sentiment_score",       0.0)
            signal.setdefault("sentiment_adjustment",  0.0)
            signal.setdefault("estimated_win_rate",    None)
            signal.setdefault("sentiment_engine",      None)

            # v2.3.0-MTF-GEMINI: new fields — default to None for historical signals
            signal.setdefault("mtf_details",      None)
            signal.setdefault("gemini_usd_bias",  None)
            signal.setdefault("gemini_engine",    None)
            # htf_confirmed was already a bool in v2.2.x — default False for older signals
            signal.setdefault("htf_confirmed",    False)

            if needs_migration:
                migrated_count += 1

        resolved = [s for s in data.get("signals", []) if s.get("status") in ("WIN", "LOSS")]
        expired  = [s for s in data.get("signals", []) if s.get("status") == "EXPIRED"]
        wins: List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics = self._empty_analytics()
        mode_stats = {
            "all":          {"trades": 0, "wins": 0, "total_pips": 0.0},
            "aggressive":   {"trades": 0, "wins": 0, "total_pips": 0.0},
            "conservative": {"trades": 0, "wins": 0, "total_pips": 0.0},
        }
        for s in resolved:
            pips       = safe_float(s.get("pips"))
            total_pips += pips
            is_win     = s["status"] == "WIN"
            if is_win: wins.append(pips)
            else:      losses.append(abs(pips))
            mode_stats["all"]["trades"] += 1
            mode_stats["all"]["total_pips"] += pips
            if is_win: mode_stats["all"]["wins"] += 1
            for mode in s.get("eligible_modes", ["aggressive"]):
                if mode in mode_stats:
                    mode_stats[mode]["trades"] += 1
                    mode_stats[mode]["total_pips"] += pips
                    if is_win: mode_stats[mode]["wins"] += 1
            self._add_to_analytics(
                analytics,
                s.get("pair", "UNKNOWN"),
                normalize_session(s.get("session")),
                normalize_confidence(s.get("confidence")),
                normalize_tier(s.get("tier")),
                s.get("eligible_modes", ["aggressive"]),
                pips, is_win,
                s.get("htf_confirmed", False),
                s.get("gemini_usd_bias")
            )
            self._add_sentiment_analytics(
                analytics, is_win, pips,
                s.get("sentiment_applied", False),
                s.get("sentiment_engine") or s.get("gemini_engine")
            )

        total      = len(resolved)
        win_rate   = (len(wins) / total * 100) if total else 0.0
        avg_win    = sum(wins) / len(wins)       if wins   else 0.0
        avg_loss   = sum(losses) / len(losses)   if losses else 0.0
        expectancy = ((win_rate/100)*avg_win - (1-win_rate/100)*avg_loss) if total else 0.0

        for md in mode_stats.values():
            if md["trades"] > 0:
                md["win_rate"] = safe_round(md["wins"] / md["trades"] * 100, 1)

        self._calculate_win_rates(analytics)

        data["analytics"] = analytics
        data["stats"] = {
            "total_trades":    total,
            "total_resolved":  total,
            "wins":            len(wins),
            "losses":          len(losses),
            "expired":         len(expired),
            "win_rate":        safe_round(win_rate, 1),
            "total_pips":      safe_round(total_pips, 1),
            "avg_win":         safe_round(avg_win, 1),
            "avg_loss":        safe_round(avg_loss, 1),
            "avg_win_pips":    safe_round(avg_win, 1),
            "avg_loss_pips":   safe_round(avg_loss, 1),
            "expectancy_pips": safe_round(expectancy, 2),
            "expectancy":      safe_round(expectancy, 2),
            "validated":       total >= 100,
            "by_mode":         mode_stats,
        }
        data["version"] = TRACKER_VERSION
        log.info(f"Migration complete: {migrated_count} signals updated")
        return data

    def _add_to_analytics(self, analytics: Dict, pair: str, session: str,
                           confidence: str, tier: str, modes: List[str],
                           pips: float, is_win: bool,
                           htf_confirmed: bool = False,
                           gemini_bias: Optional[str] = None):
        def _update(group: Dict, key: str):
            group.setdefault(key, {"trades": 0, "wins": 0, "total_pips": 0.0})
            group[key]["trades"]     += 1
            group[key]["total_pips"] += pips
            if is_win: group[key]["wins"] += 1

        _update(analytics["by_pair"],       pair)
        _update(analytics["by_session"],    session)
        _update(analytics["by_confidence"], confidence)
        _update(analytics["by_tier"],       tier)
        for mode in modes:
            _update(analytics["by_mode"], mode)

        # Tier × Session cross-analytics
        tk = f"{tier}_{session}"
        analytics["cross_analytics"]["tier_by_session"].setdefault(tk, {
            "tier": tier, "session": session, "trades": 0, "wins": 0, "total_pips": 0.0
        })
        analytics["cross_analytics"]["tier_by_session"][tk]["trades"]     += 1
        analytics["cross_analytics"]["tier_by_session"][tk]["total_pips"] += pips
        if is_win: analytics["cross_analytics"]["tier_by_session"][tk]["wins"] += 1

        # Mode × Tier cross-analytics
        for mode in modes:
            mk = f"{mode}_{tier}"
            analytics["cross_analytics"]["mode_by_tier"].setdefault(mk, {
                "mode": mode, "tier": tier, "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["cross_analytics"]["mode_by_tier"][mk]["trades"]     += 1
            analytics["cross_analytics"]["mode_by_tier"][mk]["total_pips"] += pips
            if is_win: analytics["cross_analytics"]["mode_by_tier"][mk]["wins"] += 1

        # v2.3.0: MTF confirmation analytics
        mtf_key = "fully_confirmed" if htf_confirmed else "partial_or_none"
        analytics["by_mtf_confirmed"].setdefault(mtf_key, {"trades": 0, "wins": 0, "total_pips": 0.0})
        analytics["by_mtf_confirmed"][mtf_key]["trades"]     += 1
        analytics["by_mtf_confirmed"][mtf_key]["total_pips"] += pips
        if is_win: analytics["by_mtf_confirmed"][mtf_key]["wins"] += 1

        # v2.3.0: Gemini USD bias analytics (only for USD pairs that went through the gate)
        if gemini_bias and gemini_bias in ("BULLISH", "BEARISH"):
            analytics["by_gemini_bias"].setdefault(gemini_bias, {"trades": 0, "wins": 0, "total_pips": 0.0})
            analytics["by_gemini_bias"][gemini_bias]["trades"]     += 1
            analytics["by_gemini_bias"][gemini_bias]["total_pips"] += pips
            if is_win: analytics["by_gemini_bias"][gemini_bias]["wins"] += 1

    def _add_sentiment_analytics(self, analytics: Dict, is_win: bool,
                                  pips: float, sentiment_applied: bool,
                                  sentiment_engine: Optional[str]):
        key = f"sentiment_{'on' if sentiment_applied else 'off'}"
        analytics["by_sentiment"].setdefault(key, {
            "trades": 0, "wins": 0, "total_pips": 0.0,
            "engine": sentiment_engine or "none"
        })
        analytics["by_sentiment"][key]["trades"]     += 1
        analytics["by_sentiment"][key]["total_pips"] += pips
        if is_win: analytics["by_sentiment"][key]["wins"] += 1

    def _calculate_win_rates(self, analytics: Dict):
        for group in [analytics["by_pair"], analytics["by_session"],
                      analytics["by_confidence"], analytics["by_tier"],
                      analytics["by_mode"],
                      analytics.get("by_mtf_confirmed", {}),
                      analytics.get("by_gemini_bias", {})]:
            for data in group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(data["wins"] / data["trades"] * 100, 1)

        for cross_group in analytics["cross_analytics"].values():
            for data in cross_group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(data["wins"] / data["trades"] * 100, 1)

        for data in analytics.get("by_sentiment", {}).values():
            if data["trades"] > 0:
                data["win_rate"] = safe_round(data["wins"] / data["trades"] * 100, 1)

    def _save(self):
        self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def _empty(self) -> Dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version":   TRACKER_VERSION,
            "signals":   [],
            "stats":     self._empty_stats(),
            "analytics": self._empty_analytics(),
            "daily":     {},
            "metadata":  {"created_at": now, "last_updated": now},
        }

    def _empty_stats(self) -> Dict:
        return {
            "total_trades": 0, "total_resolved": 0,
            "wins": 0, "losses": 0, "expired": 0,
            "win_rate": 0.0, "total_pips": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "avg_win_pips": 0.0, "avg_loss_pips": 0.0,
            "expectancy_pips": 0.0, "expectancy": 0.0,
            "validated": False,
            "by_mode": {
                "all":          {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0},
                "aggressive":   {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0},
                "conservative": {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0},
            },
            "note": "Stats valid after statistical significance (100+ trades)",
        }

    def _empty_analytics(self) -> Dict:
        return {
            "by_pair": {}, "by_session": {}, "by_confidence": {},
            "by_tier": {},
            "by_mode": {
                "all": {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0}
            },
            "by_sentiment": {},
            # v2.3.0 new groups
            "by_mtf_confirmed": {},
            "by_gemini_bias":   {},
            "cross_analytics": {
                "tier_by_session": {},
                "mode_by_tier":    {},
            },
        }

    def register_signal(self, signal: Dict):
        signal_id = signal.get("id") or signal.get("signal_id")
        if not signal_id:
            raise ValueError("Signal must have deterministic id or signal_id")
        if len(signal_id) < 20:
            log.warning(f"Non-deterministic signal ID: {signal_id}")
        if self._find_signal(signal_id):
            log.debug(f"Signal {signal_id} already registered"); return

        if "session"    in signal: signal["session"]    = normalize_session(signal["session"])
        if "confidence" in signal: signal["confidence"] = normalize_confidence(signal["confidence"])
        if "tier"       in signal: signal["tier"]       = normalize_tier(signal["tier"])

        signal.setdefault("eligible_modes",        ["aggressive"])
        signal.setdefault("tier",                  "C")
        signal.setdefault("sentiment_applied",     False)
        signal.setdefault("sentiment_score",       0.0)
        signal.setdefault("sentiment_adjustment",  0.0)
        signal.setdefault("sentiment_engine",      None)
        signal.setdefault("estimated_win_rate",    None)
        # v2.3.0 new fields
        signal.setdefault("htf_confirmed",    False)
        signal.setdefault("mtf_details",      None)
        signal.setdefault("gemini_usd_bias",  None)
        signal.setdefault("gemini_engine",    None)

        self.history["signals"].append(signal)
        today = datetime.now(timezone.utc).date().isoformat()
        self.history["daily"].setdefault(today, [])
        if signal_id not in self.history["daily"][today]:
            self.history["daily"][today].append(signal_id)

        self._recalculate()
        self._save()
        log.debug(f"Signal {signal_id} registered")

    def record_trade(self, signal_id: str, pair: str, direction: str,
                     entry_price: float, exit_price: float, sl: float, tp: float,
                     outcome: str, pips: float, confidence: str = None,
                     score: int = None, session: str = None,
                     entry_time: str = None, exit_time: str = None,
                     tier: str = None, eligible_modes: List[str] = None,
                     sentiment_applied: bool = False,
                     sentiment_score: float = 0.0,
                     sentiment_adjustment: float = 0.0,
                     sentiment_engine: str = None,
                     estimated_win_rate: float = None,
                     risk_reward: float = None,
                     adx: float = None,
                     rsi: float = None,
                     atr: float = None,
                     htf_confirmed: bool = False,
                     mtf_details: dict = None,
                     gemini_usd_bias: str = None,
                     gemini_engine: str = None,
                     **kwargs):
        signal = self._find_signal(signal_id)

        if signal and signal.get("status") in ("WIN", "LOSS", "EXPIRED"):
            log.warning(f"{signal_id} already resolved: {signal['status']}"); return

        if signal:
            signal["status"]     = outcome
            signal["exit_price"] = exit_price
            signal["exit_time"]  = exit_time or datetime.now(timezone.utc).isoformat()
            signal["pips"]       = pips
            if sentiment_applied:
                signal["sentiment_applied"]    = sentiment_applied
                signal["sentiment_score"]      = sentiment_score
                signal["sentiment_adjustment"] = sentiment_adjustment
                signal["sentiment_engine"]     = sentiment_engine or "finbert"
            if gemini_usd_bias is not None: signal["gemini_usd_bias"] = gemini_usd_bias
            if gemini_engine   is not None: signal["gemini_engine"]   = gemini_engine
            if mtf_details     is not None: signal["mtf_details"]     = mtf_details
            signal["htf_confirmed"] = htf_confirmed
            if risk_reward is not None: signal["risk_reward"] = risk_reward
            if adx         is not None: signal["adx"]         = adx
            if rsi         is not None: signal["rsi"]         = rsi
            if atr         is not None: signal["atr"]         = atr
            log.info(f"Updated {signal_id}: {outcome} ({pips:+.1f} pips)")
        else:
            if tier is None and score is not None:
                tier = map_score_to_tier(safe_int(score))
            signal = {
                "id":                   signal_id,
                "signal_id":            signal_id,
                "pair":                 pair,
                "direction":            direction,
                "entry_price":          entry_price,
                "exit_price":           exit_price,
                "sl":                   sl,
                "tp":                   tp,
                "status":               outcome,
                "pips":                 pips,
                "confidence":           normalize_confidence(confidence),
                "score":                score,
                "session":              normalize_session(session),
                "tier":                 normalize_tier(tier),
                "eligible_modes":       eligible_modes or map_score_to_modes(safe_int(score)),
                "sentiment_applied":    sentiment_applied,
                "sentiment_score":      sentiment_score,
                "sentiment_adjustment": sentiment_adjustment,
                "sentiment_engine":     sentiment_engine or ("finbert" if sentiment_applied else None),
                "estimated_win_rate":   estimated_win_rate,
                "risk_reward":          risk_reward,
                "adx":                  adx,
                "rsi":                  rsi,
                "atr":                  atr,
                "htf_confirmed":        htf_confirmed,
                "mtf_details":          mtf_details,
                "gemini_usd_bias":      gemini_usd_bias,
                "gemini_engine":        gemini_engine,
                "timestamp":            entry_time or datetime.now(timezone.utc).isoformat(),
                "exit_time":            exit_time  or datetime.now(timezone.utc).isoformat(),
            }
            self.history["signals"].append(signal)
            log.info(f"Recorded {signal_id}: {outcome} ({pips:+.1f} pips) [{signal['tier']}]")

        self._recalculate()
        self._save()

    def _find_signal(self, signal_id: str) -> Optional[Dict]:
        for s in self.history["signals"]:
            if s.get("id") == signal_id or s.get("signal_id") == signal_id:
                return s
        return None

    def _recalculate(self):
        resolved = [s for s in self.history["signals"] if s.get("status") in ("WIN","LOSS")]
        expired  = [s for s in self.history["signals"] if s.get("status") == "EXPIRED"]

        wins: List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics  = self._empty_analytics()
        mode_stats = {
            "all":          {"trades": 0, "wins": 0, "total_pips": 0.0},
            "aggressive":   {"trades": 0, "wins": 0, "total_pips": 0.0},
            "conservative": {"trades": 0, "wins": 0, "total_pips": 0.0},
        }

        for s in resolved:
            pips       = safe_float(s.get("pips"))
            total_pips += pips
            pair       = s.get("pair", "UNKNOWN")
            session    = normalize_session(s.get("session"))
            confidence = normalize_confidence(s.get("confidence"))
            tier       = normalize_tier(s.get("tier"))
            modes      = s.get("eligible_modes", ["aggressive"])
            is_win     = s["status"] == "WIN"
            htf        = s.get("htf_confirmed", False)
            gemini_b   = s.get("gemini_usd_bias")

            if is_win: wins.append(pips)
            else:      losses.append(abs(pips))

            mode_stats["all"]["trades"]     += 1
            mode_stats["all"]["total_pips"] += pips
            if is_win: mode_stats["all"]["wins"] += 1

            for mode in modes:
                if mode in mode_stats:
                    mode_stats[mode]["trades"]     += 1
                    mode_stats[mode]["total_pips"] += pips
                    if is_win: mode_stats[mode]["wins"] += 1

            self._add_to_analytics(analytics, pair, session, confidence, tier, modes,
                                   pips, is_win, htf, gemini_b)
            self._add_sentiment_analytics(
                analytics, is_win, pips,
                s.get("sentiment_applied", False),
                s.get("sentiment_engine") or s.get("gemini_engine")
            )

        total      = len(resolved)
        win_rate   = (len(wins) / total * 100) if total else 0.0
        avg_win    = sum(wins)   / len(wins)   if wins   else 0.0
        avg_loss   = sum(losses) / len(losses) if losses else 0.0
        expectancy = ((win_rate/100)*avg_win - (1-win_rate/100)*avg_loss) if total else 0.0

        for md in mode_stats.values():
            if md["trades"] > 0:
                md["win_rate"] = safe_round(md["wins"] / md["trades"] * 100, 1)

        self.history["stats"] = {
            "total_trades":    total,
            "total_resolved":  total,
            "wins":            len(wins),
            "losses":          len(losses),
            "expired":         len(expired),
            "win_rate":        safe_round(win_rate, 1),
            "total_pips":      safe_round(total_pips, 1),
            "avg_win":         safe_round(avg_win, 1),
            "avg_loss":        safe_round(avg_loss, 1),
            "avg_win_pips":    safe_round(avg_win, 1),
            "avg_loss_pips":   safe_round(avg_loss, 1),
            "expectancy_pips": safe_round(expectancy, 2),
            "expectancy":      safe_round(expectancy, 2),
            "validated":       total >= 100,
            "by_mode":         mode_stats,
        }

        self._calculate_win_rates(analytics)
        self.history["analytics"] = analytics

    def export_to_csv(self, path="performance_export.csv") -> str:
        pd.DataFrame(self.history["signals"]).to_csv(path, index=False)
        log.info(f"Exported to {path}")
        return path

    def get_dashboard_summary(self) -> Dict:
        return {
            "stats":          self.history["stats"],
            "analytics":      self.history["analytics"],
            "equity":         {},
            "signals_total":  len(self.history["signals"]),
            "updated_at":     datetime.now(timezone.utc).isoformat(),
            "version":        TRACKER_VERSION,
            "multi_mode":     True,
        }

    def _log_state(self):
        open_sigs = [s for s in self.history["signals"] if s.get("status") == "OPEN"]
        resolved  = [s for s in self.history["signals"] if s.get("status") in ("WIN","LOSS")]
        expired   = [s for s in self.history["signals"] if s.get("status") == "EXPIRED"]
        tier_counts: Dict[str,int] = {}
        for s in self.history["signals"]:
            t = s.get("tier","C"); tier_counts[t] = tier_counts.get(t,0)+1
        log.info(
            f"Tracker v{TRACKER_VERSION} | "
            f"Total: {len(self.history['signals'])} | Open: {len(open_sigs)} | "
            f"Resolved: {len(resolved)} | Expired: {len(expired)}"
        )
        if tier_counts:
            log.info(f"Tiers: {' | '.join(f'{t}: {c}' for t,c in sorted(tier_counts.items()))}")

        # Log MTF confirmation stats
        mtf_data = self.history.get("analytics", {}).get("by_mtf_confirmed", {})
        if mtf_data:
            fc = mtf_data.get("fully_confirmed", {})
            pc = mtf_data.get("partial_or_none", {})
            log.info(
                f"MTF Confirmed: {fc.get('trades',0)} trades {fc.get('win_rate',0):.1f}% WR | "
                f"Partial/None: {pc.get('trades',0)} trades {pc.get('win_rate',0):.1f}% WR"
            )

        # Log Gemini bias stats
        gemini_data = self.history.get("analytics", {}).get("by_gemini_bias", {})
        if gemini_data:
            for bias, v in gemini_data.items():
                log.info(f"Gemini [{bias}]: {v.get('trades',0)} trades | WR: {v.get('win_rate',0):.1f}%")

        sent = self.history.get("analytics", {}).get("by_sentiment", {})
        if sent:
            for k, v in sent.items():
                log.info(f"Sentiment [{k}]: {v.get('trades',0)} trades | WR: {v.get('win_rate',0):.1f}%")
