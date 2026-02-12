"""
Performance Tracker v2.1.3-OPTIMIZED - Aligned with Trade Beacon v2.1.3-OPTIMIZED
============================================================================

CHANGELOG v2.1.3-OPTIMIZED (based on v2.1.2-MULTI):
- ✅ Updated tier thresholds (A+: 75, A: 68, B: 60) to match v2.1.3 optimizations
- ✅ Multi-mode support: tracks eligible_modes per signal
- ✅ Tier tracking: A+, A, B, C quality classification
- ✅ Enhanced analytics: by tier, by mode, cross-analytics
- ✅ Backtest metadata: estimated_win_rate, sentiment_applied
- ✅ Mode-specific performance metrics
- ✅ Session normalization (LONDON→EUROPEAN, NEW_YORK→US)
- ✅ Confidence tier normalization (HIGH→VERY_STRONG, MEDIUM→MODERATE)
- ✅ EXPIRED signals tracking
- ✅ UTC-only datetime handling
- ✅ Deterministic ID validation
- ✅ Status transition guards
- ✅ AUTOMATIC MIGRATION: Updates old v2.1.2/v2.1.2-MULTI files to v2.1.3-OPTIMIZED on load
- ✅ Cross-analytics: tier_by_session, mode_by_tier
- ✅ Safe type conversion utilities
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd

log = logging.getLogger("performance-tracker")

TRACKER_VERSION = "2.1.3-OPTIMIZED"

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

# Normalization functions
def normalize_session(session: Optional[str]) -> str:
    """Normalize legacy session names to v2.1.3 taxonomy."""
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
    """Normalize legacy confidence levels to v2.1.3 tiers."""
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
    """Normalize tier classification."""
    if not tier:
        return "C"
    t = tier.upper()
    if t in ("A+", "A", "B", "C"):
        return t
    return "C"

def map_confidence_to_tier(confidence: str) -> str:
    """Map confidence level to tier for migration."""
    mapping = {
        'VERY_STRONG': 'A+',
        'STRONG': 'A',
        'MODERATE': 'B',
        'WEAK': 'C',
        'UNKNOWN': 'C'
    }
    return mapping.get(confidence, 'C')

def map_score_to_tier(score: int) -> str:
    """
    Map score to tier using v2.1.3-OPTIMIZED thresholds.
    UPDATED: Lowered thresholds to generate A/B tier signals.
    A+: 75+ (was 80)
    A:  68-74 (was 72-79)
    B:  60-67 (was 65-71)
    C:  below 60
    """
    if score >= 75:
        return "A+"
    elif score >= 68:
        return "A"
    elif score >= 60:
        return "B"
    else:
        return "C"

def map_score_to_modes(score: int) -> List[str]:
    """Map score to eligible modes for migration using v2.1.3 thresholds."""
    if score >= 55:
        return ['aggressive', 'conservative']
    else:
        return ['aggressive']


class PerformanceTracker:
    """
    Multi-mode signal performance tracker with tier-based analytics.

    Aligned with Trade Beacon v2.1.3-OPTIMIZED:
    - Multi-mode support: aggressive + conservative
    - Tier classification: A+, A, B, C (updated thresholds)
    - Session taxonomy: ASIAN, EUROPEAN, OVERLAP, US, LATE_US
    - Confidence tiers: VERY_STRONG, STRONG, MODERATE
    - Signal statuses: OPEN, EXPIRED, WIN, LOSS
    - UTC-only datetime handling
    - Deterministic SHA-1 signal IDs
    - Enhanced analytics: by mode, by tier, cross-analytics
    - AUTOMATIC MIGRATION from any previous version

    IMPORTANT:
    - No equity assumptions
    - No fake drawdowns
    - Stats computed from RESOLVED signals (WIN/LOSS)
    - EXPIRED signals tracked separately
    - Mode-specific performance tracking
    - Tier-based quality analysis
    """

    def __init__(self, history_file="signal_state/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.history = self._load()
        self._log_state()

    def _load(self) -> Dict:
        """Load history with automatic migration support."""
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
            self.history_file.parent.mkdir(exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)
            log.info(f"✅ Migration saved to {self.history_file}")

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
        """
        Migrate old data to v2.1.3-OPTIMIZED format.
        Handles v2.1.2, v2.1.2-MULTI, and any other previous versions.
        Re-maps tiers using updated thresholds (A+:75, A:68, B:60).
        """
        log.info(f"🔄 Migrating {len(data.get('signals', []))} signals from {from_version}...")

        migrated_count = 0
        for signal in data.get("signals", []):
            needs_migration = False

            # Normalize session
            if "session" in signal:
                old_session = signal["session"]
                signal["session"] = normalize_session(signal["session"])
                if old_session != signal["session"]:
                    needs_migration = True

            # Normalize confidence
            if "confidence" in signal:
                old_conf = signal["confidence"]
                signal["confidence"] = normalize_confidence(signal["confidence"])
                if old_conf != signal["confidence"]:
                    needs_migration = True

            # Re-map tier using updated v2.1.3 thresholds
            if "score" in signal:
                score = safe_int(signal["score"])
                new_tier = map_score_to_tier(score)
                if signal.get("tier") != new_tier:
                    signal["tier"] = new_tier
                    needs_migration = True
                    log.info(f"   ✅ Re-tiered {signal.get('pair', 'UNKNOWN')}: score={score} → {new_tier}")
            elif "tier" not in signal or not signal["tier"]:
                conf = normalize_confidence(signal.get("confidence", "MODERATE"))
                signal["tier"] = map_confidence_to_tier(conf)
                needs_migration = True

            # Add eligible_modes if missing
            if "eligible_modes" not in signal or not signal["eligible_modes"]:
                score = safe_int(signal.get("score", 0))
                signal["eligible_modes"] = map_score_to_modes(score)
                needs_migration = True
                log.info(f"   ✅ Added eligible_modes {signal['eligible_modes']} to {signal.get('pair', 'UNKNOWN')}")

            # Ensure signal_id exists
            if "signal_id" not in signal and "id" in signal:
                signal["signal_id"] = signal["id"]
                needs_migration = True

            # Add backtest/sentiment metadata if missing
            signal.setdefault("sentiment_applied", False)
            signal.setdefault("sentiment_score", 0.0)
            signal.setdefault("sentiment_adjustment", 0.0)
            signal.setdefault("estimated_win_rate", None)

            if needs_migration:
                migrated_count += 1

        # Recalculate all analytics with updated tier thresholds
        log.info(f"🔄 Recalculating analytics with v2.1.3 tier thresholds...")

        resolved = [s for s in data.get("signals", []) if s.get("status") in ("WIN", "LOSS")]
        expired = [s for s in data.get("signals", []) if s.get("status") == "EXPIRED"]

        wins: List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics = self._empty_analytics()

        mode_stats = {
            "all": {"trades": 0, "wins": 0, "total_pips": 0.0},
            "aggressive": {"trades": 0, "wins": 0, "total_pips": 0.0},
            "conservative": {"trades": 0, "wins": 0, "total_pips": 0.0}
        }

        for s in resolved:
            pips = safe_float(s.get("pips"))
            total_pips += pips
            pair = s.get("pair", "UNKNOWN")
            session = normalize_session(s.get("session"))
            confidence = normalize_confidence(s.get("confidence"))
            tier = normalize_tier(s.get("tier"))
            modes = s.get("eligible_modes", ["aggressive"])
            is_win = s["status"] == "WIN"

            if is_win:
                wins.append(pips)
            else:
                losses.append(abs(pips))

            mode_stats["all"]["trades"] += 1
            mode_stats["all"]["total_pips"] += pips
            if is_win:
                mode_stats["all"]["wins"] += 1

            for mode in modes:
                if mode in mode_stats:
                    mode_stats[mode]["trades"] += 1
                    mode_stats[mode]["total_pips"] += pips
                    if is_win:
                        mode_stats[mode]["wins"] += 1

            self._add_to_analytics(analytics, pair, session, confidence, tier, modes, pips, is_win)

        total = len(resolved)
        win_rate = (len(wins) / total * 100) if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = ((win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss) if total else 0.0

        for mode, md in mode_stats.items():
            if md["trades"] > 0:
                md["win_rate"] = safe_round(md["wins"] / md["trades"] * 100, 1)

        self._calculate_win_rates(analytics)

        data["analytics"] = analytics
        data["stats"] = {
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
            "validated": total >= 100,
            "by_mode": mode_stats
        }
        data["version"] = TRACKER_VERSION

        log.info(f"✅ Migration complete: {migrated_count} signals updated")
        log.info(f"   - by_tier: {list(analytics['by_tier'].keys())}")
        log.info(f"   - by_mode: {list(analytics['by_mode'].keys())}")

        return data

    def _add_to_analytics(self, analytics: Dict, pair: str, session: str,
                           confidence: str, tier: str, modes: List[str],
                           pips: float, is_win: bool):
        """Helper to add a resolved trade to all analytics buckets."""

        def _update(group: Dict, key: str):
            group.setdefault(key, {"trades": 0, "wins": 0, "total_pips": 0.0})
            group[key]["trades"] += 1
            group[key]["total_pips"] += pips
            if is_win:
                group[key]["wins"] += 1

        _update(analytics["by_pair"], pair)
        _update(analytics["by_session"], session)
        _update(analytics["by_confidence"], confidence)
        _update(analytics["by_tier"], tier)

        for mode in modes:
            _update(analytics["by_mode"], mode)

        # Cross analytics - tier by session
        tier_session_key = f"{tier}_{session}"
        analytics["cross_analytics"]["tier_by_session"].setdefault(tier_session_key, {
            "tier": tier, "session": session, "trades": 0, "wins": 0, "total_pips": 0.0
        })
        analytics["cross_analytics"]["tier_by_session"][tier_session_key]["trades"] += 1
        analytics["cross_analytics"]["tier_by_session"][tier_session_key]["total_pips"] += pips
        if is_win:
            analytics["cross_analytics"]["tier_by_session"][tier_session_key]["wins"] += 1

        # Cross analytics - mode by tier
        for mode in modes:
            mode_tier_key = f"{mode}_{tier}"
            analytics["cross_analytics"]["mode_by_tier"].setdefault(mode_tier_key, {
                "mode": mode, "tier": tier, "trades": 0, "wins": 0, "total_pips": 0.0
            })
            analytics["cross_analytics"]["mode_by_tier"][mode_tier_key]["trades"] += 1
            analytics["cross_analytics"]["mode_by_tier"][mode_tier_key]["total_pips"] += pips
            if is_win:
                analytics["cross_analytics"]["mode_by_tier"][mode_tier_key]["wins"] += 1

    def _calculate_win_rates(self, analytics: Dict):
        """Calculate win rates for all analytics buckets."""
        for group in [analytics["by_pair"], analytics["by_session"],
                      analytics["by_confidence"], analytics["by_tier"],
                      analytics["by_mode"]]:
            for data in group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(data["wins"] / data["trades"] * 100, 1)

        for cross_group in analytics["cross_analytics"].values():
            for data in cross_group.values():
                if data["trades"] > 0:
                    data["win_rate"] = safe_round(data["wins"] / data["trades"] * 100, 1)

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
        """Empty stats structure with multi-mode support."""
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
            "validated": False,
            "by_mode": {
                "all": {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0},
                "aggressive": {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0},
                "conservative": {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0}
            },
            "note": "Stats valid after statistical significance (100+ trades)"
        }

    def _empty_analytics(self) -> Dict:
        """Empty analytics structure with tier, mode, and cross dimensions."""
        return {
            "by_pair": {},
            "by_session": {},
            "by_confidence": {},
            "by_tier": {},
            "by_mode": {
                "all": {"trades": 0, "wins": 0, "total_pips": 0.0, "win_rate": 0.0}
            },
            "cross_analytics": {
                "tier_by_session": {},
                "mode_by_tier": {}
            }
        }

    def register_signal(self, signal: Dict):
        """
        Register a signal safely (idempotent).
        Status must be: OPEN, EXPIRED, WIN, LOSS

        Supports multi-mode signals with tier classification and sentiment data.
        """
        signal_id = signal.get("id") or signal.get("signal_id")

        if not signal_id:
            raise ValueError("Signal must have deterministic id or signal_id")

        if len(signal_id) < 20:
            log.warning(f"⚠️ Non-deterministic signal ID detected: {signal_id}")

        existing = self._find_signal(signal_id)
        if existing:
            log.debug(f"⏭️ Signal {signal_id} already registered")
            return

        # Normalize before storing
        if "session" in signal:
            signal["session"] = normalize_session(signal["session"])
        if "confidence" in signal:
            signal["confidence"] = normalize_confidence(signal["confidence"])
        if "tier" in signal:
            signal["tier"] = normalize_tier(signal["tier"])

        # Ensure multi-mode fields exist
        signal.setdefault("eligible_modes", ["aggressive"])
        signal.setdefault("tier", "C")
        signal.setdefault("sentiment_applied", False)
        signal.setdefault("sentiment_score", 0.0)
        signal.setdefault("sentiment_adjustment", 0.0)
        signal.setdefault("estimated_win_rate", None)

        self.history["signals"].append(signal)

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
                     entry_time: str = None, exit_time: str = None,
                     tier: str = None, eligible_modes: List[str] = None,
                     sentiment_applied: bool = False,
                     sentiment_score: float = 0.0,
                     sentiment_adjustment: float = 0.0,
                     estimated_win_rate: float = None,
                     **kwargs):
        """
        Record trade outcome when it hits SL/TP or expires.

        Args:
            tier: Quality tier (A+, A, B, C) - using v2.1.3 thresholds
            eligible_modes: List of modes this signal qualifies for
            sentiment_applied: Whether sentiment analysis was applied
            sentiment_score: Net sentiment value (-1.0 to +1.0)
            sentiment_adjustment: Score adjustment from sentiment
            estimated_win_rate: Micro-backtest estimated win probability
        """
        signal = self._find_signal(signal_id)

        # Status transition guard
        if signal and signal.get("status") in ("WIN", "LOSS", "EXPIRED"):
            log.warning(f"⚠️ Signal {signal_id} already resolved with status: {signal['status']}")
            return

        if signal:
            signal["status"] = outcome
            signal["exit_price"] = exit_price
            signal["exit_time"] = exit_time or datetime.now(timezone.utc).isoformat()
            signal["pips"] = pips
            log.info(f"✅ Updated signal {signal_id}: {outcome} ({pips:+.1f} pips)")
        else:
            # Re-map tier using v2.1.3 thresholds if score available
            if tier is None and score is not None:
                tier = map_score_to_tier(safe_int(score))

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
                "tier": normalize_tier(tier),
                "eligible_modes": eligible_modes or map_score_to_modes(safe_int(score)),
                "sentiment_applied": sentiment_applied,
                "sentiment_score": sentiment_score,
                "sentiment_adjustment": sentiment_adjustment,
                "estimated_win_rate": estimated_win_rate,
                "timestamp": entry_time or datetime.now(timezone.utc).isoformat(),
                "exit_time": exit_time or datetime.now(timezone.utc).isoformat()
            }
            self.history["signals"].append(signal)
            log.info(f"✅ Recorded new trade {signal_id}: {outcome} ({pips:+.1f} pips) [{signal['tier']}]")

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
        resolved = [s for s in self.history["signals"] if s.get("status") in ("WIN", "LOSS")]
        expired = [s for s in self.history["signals"] if s.get("status") == "EXPIRED"]

        wins: List[float] = []
        losses: List[float] = []
        total_pips = 0.0
        analytics = self._empty_analytics()

        mode_stats = {
            "all": {"trades": 0, "wins": 0, "total_pips": 0.0},
            "aggressive": {"trades": 0, "wins": 0, "total_pips": 0.0},
            "conservative": {"trades": 0, "wins": 0, "total_pips": 0.0}
        }

        for s in resolved:
            pips = safe_float(s.get("pips"))
            total_pips += pips
            pair = s.get("pair", "UNKNOWN")
            session = normalize_session(s.get("session"))
            confidence = normalize_confidence(s.get("confidence"))
            tier = normalize_tier(s.get("tier"))
            modes = s.get("eligible_modes", ["aggressive"])
            is_win = s["status"] == "WIN"

            if is_win:
                wins.append(pips)
            else:
                losses.append(abs(pips))

            # Track 'all' mode
            mode_stats["all"]["trades"] += 1
            mode_stats["all"]["total_pips"] += pips
            if is_win:
                mode_stats["all"]["wins"] += 1

            # Track by mode
            for mode in modes:
                if mode in mode_stats:
                    mode_stats[mode]["trades"] += 1
                    mode_stats[mode]["total_pips"] += pips
                    if is_win:
                        mode_stats[mode]["wins"] += 1

            self._add_to_analytics(analytics, pair, session, confidence, tier, modes, pips, is_win)

        total = len(resolved)
        win_rate = (len(wins) / total * 100) if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = (
            (win_rate / 100) * avg_win -
            (1 - win_rate / 100) * avg_loss
        ) if total else 0.0

        for mode, md in mode_stats.items():
            if md["trades"] > 0:
                md["win_rate"] = safe_round(md["wins"] / md["trades"] * 100, 1)

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
            "validated": total >= 100,
            "by_mode": mode_stats
        }

        self._calculate_win_rates(analytics)
        self.history["analytics"] = analytics

    def export_to_csv(self, path="performance_export.csv") -> str:
        """Export all signals to CSV with multi-mode fields."""
        pd.DataFrame(self.history["signals"]).to_csv(path, index=False)
        log.info(f"📄 Exported to {path}")
        return path

    def get_dashboard_summary(self) -> Dict:
        """Get summary for dashboard integration with multi-mode support."""
        return {
            "stats": self.history["stats"],
            "analytics": self.history["analytics"],
            "equity": {},
            "signals_total": len(self.history["signals"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "version": TRACKER_VERSION,
            "multi_mode": True
        }

    def _log_state(self):
        """Log current tracker state with multi-mode info."""
        open_sigs = [s for s in self.history["signals"] if s.get("status") == "OPEN"]
        resolved = [s for s in self.history["signals"] if s.get("status") in ("WIN", "LOSS")]
        expired = [s for s in self.history["signals"] if s.get("status") == "EXPIRED"]

        tier_counts = {}
        for s in self.history["signals"]:
            tier = s.get("tier", "C")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        log.info(
            f"📊 Tracker v{TRACKER_VERSION} Loaded | "
            f"Total: {len(self.history['signals'])} | "
            f"Open: {len(open_sigs)} | "
            f"Resolved: {len(resolved)} | "
            f"Expired: {len(expired)}"
        )

        if tier_counts:
            tier_str = " | ".join([f"{t}: {c}" for t, c in sorted(tier_counts.items())])
            log.info(f"🏆 Tiers: {tier_str}")
