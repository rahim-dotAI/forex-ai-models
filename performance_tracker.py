"""
Performance Tracker v2.4.0 — Trade Beacon FOCUSED GBP EDITION
============================================================================

CHANGELOG v2.4.0:
- Version bumped to 2.4.0
- GBPAUD / GBPCAD noted in pair analytics as permanently blocked
- by_mtf_confirmed analytics retained (tracks fully_confirmed vs partial)
- by_gemini_bias analytics retained (tracks Gemini USD gate effectiveness)
- Migration: adds mtf_confirmed field alongside htf_confirmed for v2.3.x signals

CHANGELOG v2.3.0-MTF-GEMINI (carried forward):
- mtf_details, gemini_usd_bias, gemini_engine fields added to signal schema
- by_mtf_confirmed analytics group (WR when all TFs confirmed vs partial)
- by_gemini_bias analytics group (WR when Gemini said BULLISH vs BEARISH)
- Auto-migration: defaults new fields to None/False for all historical signals

CHANGELOG v2.2.2-SAFE (carried forward):
- avg_win_pips / avg_loss_pips aliases
- expectancy alias alongside expectancy_pips
- Tier thresholds: A+:80, A:72, B:55
- Resolve dedup guard
- cross_analytics: tier_by_session + mode_by_tier
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd

log = logging.getLogger("performance-tracker")

TRACKER_VERSION = "2.4.0"

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
    return {"VERY_STRONG":"A+","STRONG":"A","MODERATE":"B","WEAK":"C","UNKNOWN":"C"}.get(confidence,"C")

def map_score_to_tier(score: int) -> str:
    """v2.4.0 tier thresholds: A+≥80, A≥72 (blocked), B≥55, C<55"""
    if score >= 80: return "A+"
    if score >= 72: return "A"
    if score >= 55: return "B"
    return "C"

def map_score_to_modes(score: int) -> List[str]:
    if score >= 57: return ["aggressive", "conservative"]
    if score >= 55: return ["aggressive"]
    return []


class PerformanceTracker:
    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load()
        self._log_state()

    # ─────────────────────────────────────────────────────────────────────────
    # LOAD / MIGRATE
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> Dict:
        if not self.history_file.exists():
            return self._empty()
        try:
            with open(self.history_file) as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"Could not load history: {e} — starting fresh")
            return self._empty()

        old_version = data.get("version", "unknown")
        if old_version != TRACKER_VERSION:
            log.info(f"Migrating history from {old_version} → {TRACKER_VERSION}")
            data = self._migrate(data, old_version)
            try:
                with open(self.history_file, "w") as f:
                    json.dump(data, f, indent=2)
                log.info("Migration saved")
            except Exception as e:
                log.warning(f"Could not save migrated history: {e}")

        data.setdefault("version",   TRACKER_VERSION)
        data.setdefault("signals",   [])
        data.setdefault("stats",     self._empty_stats())
        data.setdefault("analytics", self._empty_analytics())
        data.setdefault("daily",     {})
        data.setdefault("metadata",  {
            "created_at":   datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })
        return data

    def _migrate(self, data: Dict, from_version: str) -> Dict:
        signals = data.get("signals", [])
        log.info(f"Migrating {len(signals)} signals from {from_version}…")
        migrated = 0

        for s in signals:
            changed = False

            # Session / confidence normalization
            if "session" in s:
                norm = normalize_session(s["session"])
                if s["session"] != norm:
                    s["session"] = norm; changed = True
            if "confidence" in s:
                norm = normalize_confidence(s["confidence"])
                if s["confidence"] != norm:
                    s["confidence"] = norm; changed = True

            # Tier from score
            if "score" in s:
                nt = map_score_to_tier(safe_int(s["score"]))
                if s.get("tier") != nt:
                    s["tier"] = nt; changed = True
            elif not s.get("tier"):
                s["tier"] = map_confidence_to_tier(normalize_confidence(s.get("confidence","MODERATE")))
                changed = True

            # Eligible modes
            if not s.get("eligible_modes"):
                s["eligible_modes"] = map_score_to_modes(safe_int(s.get("score",0)))
                changed = True

            # signal_id alias
            if "signal_id" not in s and "id" in s:
                s["signal_id"] = s["id"]; changed = True

            # v2.2.2-SAFE fields
            s.setdefault("sentiment_applied",    False)
            s.setdefault("sentiment_score",      0.0)
            s.setdefault("sentiment_adjustment", 0.0)
            s.setdefault("estimated_win_rate",   None)
            s.setdefault("sentiment_engine",     None)

            # v2.3.0 fields
            s.setdefault("htf_confirmed",   False)
            s.setdefault("mtf_details",     None)
            s.setdefault("gemini_usd_bias", None)
            s.setdefault("gemini_engine",   None)

            # v2.4.0: mtf_confirmed alias (same as htf_confirmed)
            s.setdefault("mtf_confirmed", s.get("htf_confirmed", False))

            if changed:
                migrated += 1

        # Rebuild all stats and analytics from migrated signals
        resolved = [s for s in signals if s.get("status") in ("WIN","LOSS")]
        expired  = [s for s in signals if s.get("status") == "EXPIRED"]

        wins:   List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics  = self._empty_analytics()
        mode_stats = self._empty_mode_stats()

        for s in resolved:
            pips   = safe_float(s.get("pips"))
            is_win = s["status"] == "WIN"
            total_pips += pips
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
                s.get("pair","UNKNOWN"),
                normalize_session(s.get("session")),
                normalize_confidence(s.get("confidence")),
                normalize_tier(s.get("tier")),
                s.get("eligible_modes",["aggressive"]),
                pips, is_win,
                s.get("mtf_confirmed", s.get("htf_confirmed", False)),
                s.get("gemini_usd_bias"),
            )
            self._add_sentiment_analytics(
                analytics, is_win, pips,
                s.get("sentiment_applied", False),
                s.get("sentiment_engine") or s.get("gemini_engine"),
            )

        total      = len(resolved)
        win_rate   = (len(wins)/total*100) if total else 0.0
        avg_win    = sum(wins)/len(wins)     if wins   else 0.0
        avg_loss   = sum(losses)/len(losses) if losses else 0.0
        expectancy = ((win_rate/100)*avg_win - (1-win_rate/100)*avg_loss) if total else 0.0

        for md in mode_stats.values():
            if md["trades"] > 0:
                md["win_rate"] = safe_round(md["wins"]/md["trades"]*100, 1)

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
        log.info(f"Migration complete: {migrated} signals updated, stats rebuilt")
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYTICS HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _add_to_analytics(self, analytics: Dict, pair: str, session: str,
                           confidence: str, tier: str, modes: List[str],
                           pips: float, is_win: bool,
                           mtf_confirmed: bool = False,
                           gemini_bias: Optional[str] = None):
        def _upd(group: Dict, key: str):
            group.setdefault(key, {"trades":0,"wins":0,"total_pips":0.0})
            group[key]["trades"]     += 1
            group[key]["total_pips"] += pips
            if is_win: group[key]["wins"] += 1

        _upd(analytics["by_pair"],       pair)
        _upd(analytics["by_session"],    session)
        _upd(analytics["by_confidence"], confidence)
        _upd(analytics["by_tier"],       tier)
        for m in modes:
            _upd(analytics["by_mode"], m)

        # Tier × Session
        tk = f"{tier}_{session}"
        analytics["cross_analytics"]["tier_by_session"].setdefault(tk, {
            "tier":tier,"session":session,"trades":0,"wins":0,"total_pips":0.0
        })
        analytics["cross_analytics"]["tier_by_session"][tk]["trades"]     += 1
        analytics["cross_analytics"]["tier_by_session"][tk]["total_pips"] += pips
        if is_win: analytics["cross_analytics"]["tier_by_session"][tk]["wins"] += 1

        # Mode × Tier
        for m in modes:
            mk = f"{m}_{tier}"
            analytics["cross_analytics"]["mode_by_tier"].setdefault(mk, {
                "mode":m,"tier":tier,"trades":0,"wins":0,"total_pips":0.0
            })
            analytics["cross_analytics"]["mode_by_tier"][mk]["trades"]     += 1
            analytics["cross_analytics"]["mode_by_tier"][mk]["total_pips"] += pips
            if is_win: analytics["cross_analytics"]["mode_by_tier"][mk]["wins"] += 1

        # MTF confirmation
        mtf_key = "fully_confirmed" if mtf_confirmed else "partial_or_none"
        analytics["by_mtf_confirmed"].setdefault(mtf_key, {"trades":0,"wins":0,"total_pips":0.0})
        analytics["by_mtf_confirmed"][mtf_key]["trades"]     += 1
        analytics["by_mtf_confirmed"][mtf_key]["total_pips"] += pips
        if is_win: analytics["by_mtf_confirmed"][mtf_key]["wins"] += 1

        # Gemini USD bias (only for signals that went through the gate)
        if gemini_bias and gemini_bias in ("BULLISH","BEARISH"):
            analytics["by_gemini_bias"].setdefault(gemini_bias, {"trades":0,"wins":0,"total_pips":0.0})
            analytics["by_gemini_bias"][gemini_bias]["trades"]     += 1
            analytics["by_gemini_bias"][gemini_bias]["total_pips"] += pips
            if is_win: analytics["by_gemini_bias"][gemini_bias]["wins"] += 1

    def _add_sentiment_analytics(self, analytics: Dict, is_win: bool, pips: float,
                                  sentiment_applied: bool, engine: Optional[str]):
        key = f"sentiment_{'on' if sentiment_applied else 'off'}"
        analytics["by_sentiment"].setdefault(key, {
            "trades":0,"wins":0,"total_pips":0.0,"engine": engine or "none"
        })
        analytics["by_sentiment"][key]["trades"]     += 1
        analytics["by_sentiment"][key]["total_pips"] += pips
        if is_win: analytics["by_sentiment"][key]["wins"] += 1

    def _calculate_win_rates(self, analytics: Dict):
        for group in [
            analytics["by_pair"], analytics["by_session"],
            analytics["by_confidence"], analytics["by_tier"], analytics["by_mode"],
            analytics.get("by_mtf_confirmed",{}), analytics.get("by_gemini_bias",{}),
        ]:
            for d in group.values():
                if d.get("trades",0) > 0:
                    d["win_rate"] = safe_round(d["wins"]/d["trades"]*100, 1)

        for cross_group in analytics["cross_analytics"].values():
            for d in cross_group.values():
                if d.get("trades",0) > 0:
                    d["win_rate"] = safe_round(d["wins"]/d["trades"]*100, 1)

        for d in analytics.get("by_sentiment",{}).values():
            if d.get("trades",0) > 0:
                d["win_rate"] = safe_round(d["wins"]/d["trades"]*100, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # EMPTY TEMPLATES
    # ─────────────────────────────────────────────────────────────────────────

    def _empty_mode_stats(self) -> Dict:
        return {
            "all":          {"trades":0,"wins":0,"total_pips":0.0,"win_rate":0.0},
            "aggressive":   {"trades":0,"wins":0,"total_pips":0.0,"win_rate":0.0},
            "conservative": {"trades":0,"wins":0,"total_pips":0.0,"win_rate":0.0},
        }

    def _empty_stats(self) -> Dict:
        return {
            "total_trades":0,"total_resolved":0,
            "wins":0,"losses":0,"expired":0,
            "win_rate":0.0,"total_pips":0.0,
            "avg_win":0.0,"avg_loss":0.0,
            "avg_win_pips":0.0,"avg_loss_pips":0.0,
            "expectancy_pips":0.0,"expectancy":0.0,
            "validated":False,
            "by_mode": self._empty_mode_stats(),
            "_note": "Stats valid after 100+ trades (statistical significance threshold).",
        }

    def _empty_analytics(self) -> Dict:
        return {
            "by_pair":{}, "by_session":{}, "by_confidence":{},
            "by_tier":{},
            "by_mode": {
                "all": {"trades":0,"wins":0,"total_pips":0.0,"win_rate":0.0}
            },
            "by_sentiment":       {},
            "by_mtf_confirmed":   {},  # fully_confirmed vs partial_or_none
            "by_gemini_bias":     {},  # BULLISH vs BEARISH
            "cross_analytics": {
                "tier_by_session": {},
                "mode_by_tier":    {},
            },
        }

    def _empty(self) -> Dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version":   TRACKER_VERSION,
            "signals":   [],
            "stats":     self._empty_stats(),
            "analytics": self._empty_analytics(),
            "daily":     {},
            "metadata":  {"created_at":now,"last_updated":now},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        self.history["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def register_signal(self, signal: Dict):
        sid = signal.get("id") or signal.get("signal_id")
        if not sid:
            raise ValueError("Signal must have a deterministic id / signal_id")
        if self._find_signal(sid):
            log.debug(f"Signal {sid} already registered"); return

        if "session"    in signal: signal["session"]    = normalize_session(signal["session"])
        if "confidence" in signal: signal["confidence"] = normalize_confidence(signal["confidence"])
        if "tier"       in signal: signal["tier"]       = normalize_tier(signal["tier"])

        signal.setdefault("eligible_modes",       ["aggressive"])
        signal.setdefault("tier",                 "C")
        signal.setdefault("sentiment_applied",    False)
        signal.setdefault("sentiment_score",      0.0)
        signal.setdefault("sentiment_adjustment", 0.0)
        signal.setdefault("sentiment_engine",     None)
        signal.setdefault("estimated_win_rate",   None)
        # v2.3.0+ fields
        signal.setdefault("htf_confirmed",    False)
        signal.setdefault("mtf_confirmed",    False)
        signal.setdefault("mtf_details",      None)
        signal.setdefault("gemini_usd_bias",  None)
        signal.setdefault("gemini_engine",    None)

        self.history["signals"].append(signal)
        today = datetime.now(timezone.utc).date().isoformat()
        self.history["daily"].setdefault(today, [])
        if sid not in self.history["daily"][today]:
            self.history["daily"][today].append(sid)

        self._recalculate()
        self._save()
        log.debug(f"Signal {sid} registered")

    def record_trade(self, signal_id: str, pair: str, direction: str,
                     entry_price: float, exit_price: float, sl: float, tp: float,
                     outcome: str, pips: float,
                     confidence: str = None, score: int = None, session: str = None,
                     entry_time: str = None, exit_time: str = None,
                     tier: str = None, eligible_modes: List[str] = None,
                     sentiment_applied: bool = False,
                     sentiment_score: float = 0.0,
                     sentiment_adjustment: float = 0.0,
                     sentiment_engine: str = None,
                     estimated_win_rate: float = None,
                     risk_reward: float = None,
                     adx: float = None, rsi: float = None, atr: float = None,
                     htf_confirmed: bool = False,
                     mtf_confirmed: bool = False,
                     mtf_details: dict = None,
                     gemini_usd_bias: str = None,
                     gemini_engine: str = None,
                     **kwargs):
        signal = self._find_signal(signal_id)

        if signal and signal.get("status") in ("WIN","LOSS","EXPIRED"):
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
            signal["htf_confirmed"]  = htf_confirmed
            signal["mtf_confirmed"]  = mtf_confirmed or htf_confirmed
            if risk_reward is not None: signal["risk_reward"] = risk_reward
            if adx is not None: signal["adx"] = adx
            if rsi is not None: signal["rsi"] = rsi
            if atr is not None: signal["atr"] = atr
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
                "mtf_confirmed":        mtf_confirmed or htf_confirmed,
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

        wins:   List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics  = self._empty_analytics()
        mode_stats = self._empty_mode_stats()

        for s in resolved:
            pips       = safe_float(s.get("pips"))
            total_pips += pips
            is_win     = s["status"] == "WIN"
            if is_win: wins.append(pips)
            else:      losses.append(abs(pips))

            mode_stats["all"]["trades"]     += 1
            mode_stats["all"]["total_pips"] += pips
            if is_win: mode_stats["all"]["wins"] += 1
            for m in s.get("eligible_modes",["aggressive"]):
                if m in mode_stats:
                    mode_stats[m]["trades"]     += 1
                    mode_stats[m]["total_pips"] += pips
                    if is_win: mode_stats[m]["wins"] += 1

            self._add_to_analytics(
                analytics,
                s.get("pair","UNKNOWN"),
                normalize_session(s.get("session")),
                normalize_confidence(s.get("confidence")),
                normalize_tier(s.get("tier")),
                s.get("eligible_modes",["aggressive"]),
                pips, is_win,
                s.get("mtf_confirmed", s.get("htf_confirmed", False)),
                s.get("gemini_usd_bias"),
            )
            self._add_sentiment_analytics(
                analytics, is_win, pips,
                s.get("sentiment_applied",False),
                s.get("sentiment_engine") or s.get("gemini_engine"),
            )

        total      = len(resolved)
        win_rate   = (len(wins)/total*100) if total else 0.0
        avg_win    = sum(wins)/len(wins)     if wins   else 0.0
        avg_loss   = sum(losses)/len(losses) if losses else 0.0
        expectancy = ((win_rate/100)*avg_win - (1-win_rate/100)*avg_loss) if total else 0.0

        for md in mode_stats.values():
            if md["trades"] > 0:
                md["win_rate"] = safe_round(md["wins"]/md["trades"]*100, 1)

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
            "stats":         self.history["stats"],
            "analytics":     self.history["analytics"],
            "equity":        {},
            "signals_total": len(self.history["signals"]),
            "updated_at":    datetime.now(timezone.utc).isoformat(),
            "version":       TRACKER_VERSION,
            "multi_mode":    True,
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
            log.info(f"Tiers: {' | '.join(f'{t}:{c}' for t,c in sorted(tier_counts.items()))}")

        # MTF stats
        mtf = self.history.get("analytics",{}).get("by_mtf_confirmed",{})
        if mtf:
            fc = mtf.get("fully_confirmed",{}); pc = mtf.get("partial_or_none",{})
            log.info(
                f"MTF Confirmed: {fc.get('trades',0)} trades {fc.get('win_rate',0):.1f}% WR | "
                f"Partial/None: {pc.get('trades',0)} trades {pc.get('win_rate',0):.1f}% WR"
            )

        # Gemini USD stats
        gemini = self.history.get("analytics",{}).get("by_gemini_bias",{})
        for bias, v in gemini.items():
            log.info(f"Gemini [{bias}]: {v.get('trades',0)} trades | WR: {v.get('win_rate',0):.1f}%")

        # Sentiment breakdown
        sent = self.history.get("analytics",{}).get("by_sentiment",{})
        for k, v in sent.items():
            log.info(f"Sentiment [{k}]: {v.get('trades',0)} trades | WR: {v.get('win_rate',0):.1f}%")
