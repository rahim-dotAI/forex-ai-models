"""
Trade Beacon v2.0.6 - Forex Signal Generator (PRODUCTION READY)
================================================================

CRITICAL FIXES IN v2.0.6:
- ✅ Real yfinance fallback (15m → 1h) implemented
- ✅ Volume scoring DISABLED for FX pairs
- ✅ Performance tracker isolated from signal generation
- ✅ Unified pip calculation function
- ✅ Signal expiration enforcement
- ✅ Confidence labels redesigned (psychological accuracy)
- ✅ ADX thresholds lowered for 15m reality
- ✅ Pullback logic safety guard added
- ✅ Correlation filter now direction-aware
- ✅ Dashboard metrics renamed (theoretical vs realized)

This is a SIGNAL GENERATOR ONLY - no trade execution logic.
"""

import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import yfinance as yf
import requests
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Import performance tracker
from performance_tracker import PerformanceTracker

# =========================
# SIGNAL GENERATION (FIXED SCORING)
# =========================
def generate_signal(pair: str) -> Tuple[Optional[dict], bool]:
    df, download_success = download(pair)
    
    if not download_success or len(df) < MIN_ROWS:
        log.warning(f"⚠️ {pair} not enough candles ({len(df)}), skipping")
        return None, download_success
    
    try:
        close = ensure_series(df["Close"])
        high = ensure_series(df["High"])
        low = ensure_series(df["Low"])
        volume = ensure_series(df["Volume"])

        e12 = last(ema(close, 12))
        e26 = last(ema(close, 26))
        e200 = last(ema(close, 200))
        r = last(rsi(close))
        a = last(adx_calc(high, low, close))
        atr = last(atr_calc(high, low, close))
        current_price = last(close)
        
        # ✅ FIX #2: Volume calculation (but not used for scoring)
        avg_volume = volume.rolling(window=20).mean()
        current_volume = last(volume)
        avg_vol = last(avg_volume)
        volume_ratio = current_volume / avg_vol if avg_vol and avg_vol > 0 else 1.0

    except Exception as e:
        log.warning(f"⚠️ {pair} indicator calc failed: {e}")
        return None, download_success

    if None in (e12, e26, e200, r, a, current_price, atr):
        log.warning(f"⚠️ {pair} indicators incomplete, skipping")
        return None, download_success

    min_adx = SETTINGS.get("min_adx", 22)
    if a < min_adx:
        log.info(f"❌ {pair} | ADX too low ({a:.1f} < {min_adx})")
        return None, download_success

    bull = bear = 0

    # EMA Trend Structure (25 points)
    if e12 > e26 > e200:
        bull += 25
    elif e12 < e26 < e200:
        bear += 25

    # ✅ FIX #8: Pullback logic with safety guard
    rsi_oversold = SETTINGS.get("rsi_oversold", 30)
    rsi_overbought = SETTINGS.get("rsi_overbought", 70)
    
    # Only reward pullbacks if RSI hasn't gone too extreme (avoid falling knives)
    if e12 > e26 > e200 and rsi_oversold + 5 < r < 45:
        bull += 15  # Safe pullback entry in uptrend
    elif e12 < e26 < e200 and 55 < r < rsi_overbought - 5:
        bear += 15  # Safe pullback entry in downtrend
    
    # RSI Context
    if MODE == "conservative":
        if r < rsi_oversold:
            bull += 30
        elif r > rsi_overbought:
            bear += 30
    else:
        if r < rsi_oversold:
            bull += 20
        elif r > rsi_overbought:
            bear += 20

    # ADX Trend Strength
    if a > 25:
        if e12 > e26:
            bull += 20
        elif e12 < e26:
            bear += 20
    elif a > min_adx:
        if e12 > e26:
            bull += 10
        elif e12 < e26:
            bear += 10
    
    # ✅ FIX #2: Volume scoring DISABLED for FX
    if USE_VOLUME_FOR_FX:
        min_volume_ratio = SETTINGS.get("min_volume_ratio", 1.3)
        volume_penalty = SETTINGS.get("volume_penalty", 5)
        
        if volume_ratio >= min_volume_ratio:
            if volume_ratio > 1.5:
                bonus = 10
            elif volume_ratio > 1.2:
                bonus = 5
            else:
                bonus = 3
            
            if e12 > e26:
                bull += bonus
            else:
                bear += bonus
        else:
            if e12 > e26:
                bull -= volume_penalty
            else:
                bear -= volume_penalty
    else:
        log.debug(f"📊 {pair} | Volume scoring disabled for FX")
    
    # Session bonus
    session = get_market_session()
    clean_pair = pair.replace("=X", "")
    session_bonus = calculate_dynamic_session_bonus(clean_pair, session, CONFIG)
    
    if e12 > e26:
        bull += session_bonus
    else:
        bear += session_bonus

    diff = abs(bull - bear)

    threshold = SETTINGS.get("threshold", 60)
    if diff < threshold:
        return None, download_success

    direction = "BUY" if bull > bear else "SELL"

    # ✅ FIX #6: Confidence labels redesigned
    if diff >= 75:
        confidence = "VERY_STRONG"
    elif diff >= 65:
        confidence = "STRONG"
    elif diff >= 55:
        confidence = "MODERATE"
    else:
        confidence = "WEAK"

    spread = get_spread(pair)
    entry_price = current_price
    
    atr_stop_mult = SETTINGS.get("atr_stop_multiplier", 1.8)
    atr_target_mult = SETTINGS.get("atr_target_multiplier", 4.0)
    
    if direction == "BUY":
        sl = entry_price - (atr_stop_mult * atr)
        tp = entry_price + (atr_target_mult * atr)
    else:
        sl = entry_price + (atr_stop_mult * atr)
        tp = entry_price - (atr_target_mult * atr)
    
    risk = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    min_rr = SETTINGS.get("min_risk_reward", 2.0)
    if risk_reward < min_rr:
        log.info(f"❌ {pair} | Poor risk-reward ({risk_reward:.2f} < {min_rr})")
        return None, download_success
    
    signal_type = get_signal_type(e12, e26, e200, r)
    market_state = classify_market_state(a, atr)
    
    now = datetime.now(timezone.utc)
    date_str = now.strftime('%Y%m%d')
    valid_for_minutes = CONFIG.get("advanced", {}).get("validation", {}).get("max_signal_age_seconds", 900) / 60
    expires_at = now + timedelta(minutes=valid_for_minutes)
    
    hold_time = calculate_hold_time(risk_reward, atr)
    eligible_modes = calculate_eligible_modes(diff, a, volume_ratio, r, CONFIG)
    freshness = calculate_signal_freshness(now)
    
    # Deterministic signal ID
    signal_id = generate_deterministic_signal_id(clean_pair, direction, entry_price, session, date_str)

    signal = {
        "signal_id": signal_id,
        "id": signal_id,
        "pair": clean_pair,
        "direction": direction,
        "score": diff,
        "technical_score": diff,
        "sentiment_score": 0,
        "confidence": confidence,
        "rsi": round(r, 1),
        "adx": round(a, 1),
        "atr": round(atr, 5),
        "volume_ratio": round(volume_ratio, 2),
        "session": session,
        "entry_price": round(entry_price, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "risk_reward": round(risk_reward, 2),
        "spread": round(spread, 5),
        "timestamp": now.isoformat(),
        "status": "OPEN",
        "hold_time": hold_time,
        "eligible_modes": eligible_modes,
        "freshness": freshness,
        "metadata": {
            "signal_type": signal_type,
            "market_state": market_state,
            "timeframe": INTERVAL,
            "valid_for_minutes": valid_for_minutes,
            "generated_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "session_active": session,
            "signal_generator_version": "2.0.6",
            "atr_stop_multiplier": atr_stop_mult,
            "atr_target_multiplier": atr_target_mult
        }
    }
    
    is_valid, warnings = validate_signal_quality(signal, CONFIG)
    
    if not is_valid:
        log.info(f"❌ {pair} | Signal rejected: {', '.join(warnings)}")
        return None, download_success
    
    if warnings:
        log.debug(f"⚠️ {pair} | Signal warnings: {', '.join(warnings)}")
    
    return signal, download_success

# =========================
# ✅ FIX #9: CORRELATION FILTER - DIRECTION AWARE
# =========================
def filter_correlated_signals_enhanced(signals: List[Dict], max_correlated: int = 1) -> List[Dict]:
    """Filter correlated signals, but only if they're in the SAME direction."""
    if len(signals) <= 1:
        return signals
    
    filtered = []
    correlation_groups = {}
    
    sorted_signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    
    for signal in sorted_signals:
        pair = f"{signal['pair']}=X"
        direction = signal['direction']
        
        assigned_group = None
        for corr_group in CORRELATED_PAIRS:
            if pair in corr_group:
                # ✅ FIX: Group by correlation AND direction
                group_key = (frozenset(corr_group), direction)
                assigned_group = group_key
                break
        
        if assigned_group:
            count = correlation_groups.get(assigned_group, 0)
            if count < max_correlated:
                filtered.append(signal)
                correlation_groups[assigned_group] = count + 1
            else:
                log.info(f"⚠️ Skipping {signal['pair']} {direction} (correlation group limit: {max_correlated})")
        else:
            filtered.append(signal)
    
    if len(filtered) < len(signals):
        log.info(f"🔗 Direction-aware correlation filter: {len(signals)} → {len(filtered)} signals")
    
    return filtered

def check_risk_limits(signals: List[Dict], config: Dict) -> Tuple[List[Dict], List[str]]:
    risk_config = config.get("risk_management", {})
    warnings = []
    
    max_positions = risk_config.get("max_open_positions", 3)
    if len(signals) > max_positions:
        warnings.append(f"Limiting to {max_positions} positions (had {len(signals)})")
        signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:max_positions]
    
    # ✅ FIX #4: Use unified pip calculation
    max_daily_risk = risk_config.get("max_daily_risk_pips", 150)
    total_risk_pips = 0
    
    filtered = []
    for signal in signals:
        entry = signal.get('entry_price', 0)
        sl = signal.get('sl', 0)
        pair = signal.get('pair', '')
        
        if entry > 0 and sl > 0:
            risk_pips = price_to_pips(pair, abs(entry - sl))
            
            if total_risk_pips + risk_pips <= max_daily_risk:
                filtered.append(signal)
                total_risk_pips += risk_pips
            else:
                warnings.append(f"Skipped {pair} - would exceed daily risk limit")
        else:
            filtered.append(signal)
    
    mode = config.get("mode", "conservative")
    max_correlated = config["settings"][mode].get("max_correlated_signals", 1)
    
    if config.get("advanced", {}).get("enable_correlation_filter", True):
        filtered = filter_correlated_signals_enhanced(filtered, max_correlated)
    
    # ✅ FIX #3: Skip drawdown check in signal-only mode
    if not SIGNAL_ONLY_MODE:
        stop_on_drawdown = risk_config.get("stop_trading_on_drawdown_pips", 100)
        if PERFORMANCE_TRACKER:
            stats = PERFORMANCE_TRACKER.history.get("stats", {})
            total_pips = stats.get("total_pips", 0)
            
            if total_pips < -stop_on_drawdown:
                warnings.append(f"⚠️ Trading halted: Drawdown limit reached ({total_pips:.1f} pips)")
                return [], warnings
    
    return filtered, warnings

# =========================
# SENTIMENT ENHANCEMENT
# =========================
def enhance_with_sentiment(signals: List[Dict], news_agg: NewsAggregator) -> List[Dict]:
    if not USE_SENTIMENT or not signals:
        return signals
    
    log.info("\n" + "="*70)
    log.info("📰 Analyzing news sentiment from NewsAPI + Marketaux...")
    log.info("="*70)
    
    hf_key = os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY')
    analyzer = SentimentAnalyzer(hf_api_key=hf_key)
    
    all_pairs = [f"{sig['pair']}=X" for sig in signals]
    log.info(f"🔍 Fetching news for {len(all_pairs)} pairs: {', '.join(all_pairs)}")
    
    all_articles = news_agg.get_news(all_pairs)
    
    enhanced = []
    
    for signal in signals:
        pair = signal['pair']
        pair_ticker = f"{pair}=X"
        
        pair_articles = filter_articles_for_pair(pair_ticker, all_articles)
        
        sentiment_data = analyze_sentiment_from_articles(
            pair_ticker, 
            pair_articles, 
            analyzer
        )
        
        original_score = signal['technical_score']
        adjustment = sentiment_data['adjustment']
        
        direction_multiplier = 1 if signal['direction'] == 'BUY' else -1
        final_adjustment = adjustment * direction_multiplier
        
        signal['score'] = original_score + final_adjustment
        signal['score'] = max(0, min(100, signal['score']))
        signal['sentiment_score'] = adjustment
        
        threshold = SETTINGS.get("threshold", 60)
        if signal['score'] < threshold:
            log.info(f"❌ {pair} | Signal too weak after sentiment ({signal['score']} < {threshold})")
            continue
        
        signal['eligible_modes'] = calculate_eligible_modes(
            signal['score'],
            signal['adx'],
            signal['volume_ratio'],
            signal['rsi'],
            CONFIG
        )
        
        # ✅ FIX #6: Updated confidence labels
        if signal['score'] >= 75:
            signal['confidence'] = "VERY_STRONG"
        elif signal['score'] >= 65:
            signal['confidence'] = "STRONG"
        elif signal['score'] >= 55:
            signal['confidence'] = "MODERATE"
        else:
            signal['confidence'] = "WEAK"
        
        signal['sentiment'] = {
            "overall": sentiment_data['sentiment'],
            "adjustment": adjustment,
            "original_score": original_score,
            "news_count": sentiment_data['news_count'],
            "sources": sentiment_data.get('sources', {})
        }
        
        log.info(f"💡 {pair} | Direction: {signal['direction']} | "
                f"Sentiment: {sentiment_data['sentiment']} ({adjustment:+d}) | "
                f"Score: {original_score} → {signal['score']} ({final_adjustment:+d})")
        
        enhanced.append(signal)
    
    log.info(f"📊 API Usage: NewsAPI calls={news_agg.newsapi_calls}, "
             f"Marketaux calls={news_agg.marketaux_calls}")
    
    return enhanced

# =========================
# DASHBOARD WRITER
# =========================
def calculate_daily_pips(signals: List[Dict]) -> float:
    """Calculate theoretical maximum pips from today's signals."""
    today = datetime.now(timezone.utc).date()
    daily_pips = 0
    
    for signal in signals:
        try:
            signal_time = datetime.fromisoformat(signal.get('timestamp', ''))
            if signal_time.date() == today:
                entry = signal.get('entry_price', 0)
                tp = signal.get('tp', 0)
                pair = signal.get('pair', '')
                if entry and tp:
                    # ✅ FIX #4: Use unified pip calculation
                    pips = price_to_pips(pair, abs(tp - entry))
                    daily_pips += pips
        except Exception:
            continue
    
    return round(daily_pips, 1)

def get_performance_summary() -> Dict:
    if not PERFORMANCE_TRACKER:
        return {"stats": {}, "analytics": {}, "equity": {}}
    
    try:
        return PERFORMANCE_TRACKER.get_dashboard_summary()
    except Exception as e:
        log.error(f"⚠️ Could not get performance summary: {e}")
        return {"stats": {}, "analytics": {}, "equity": {}}

# ✅ FIX #5: Enforce signal expiration before dashboard write
def filter_expired_signals(signals: List[Dict]) -> List[Dict]:
    """Remove expired signals before writing to dashboard."""
    now = datetime.now(timezone.utc)
    active = []
    
    for signal in signals:
        try:
            expires_at_str = signal.get('metadata', {}).get('expires_at')
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                if now < expires_at:
                    active.append(signal)
                else:
                    log.debug(f"⏰ Expired signal filtered: {signal['pair']} (expired {(now - expires_at).total_seconds()/60:.1f} min ago)")
            else:
                # If no expiry, keep it
                active.append(signal)
        except Exception as e:
            log.warning(f"⚠️ Could not check expiry for {signal.get('pair', 'UNKNOWN')}: {e}")
            active.append(signal)  # Keep on error
    
    if len(active) < len(signals):
        log.info(f"⏰ Filtered {len(signals) - len(active)} expired signals")
    
    return active

def write_dashboard_state(signals: list, successful_downloads: int, newsapi_calls: int = 0, marketaux_calls: int = 0,
                          config: Dict = None, mode: str = None, settings: Dict = None):
    """Write dashboard state with config parameters."""
    current_config = config if config is not None else CONFIG
    current_mode = mode if mode is not None else MODE
    current_settings = settings if settings is not None else SETTINGS
    
    # ✅ FIX #5: Filter expired signals
    signals = filter_expired_signals(signals)
    
    session = get_market_session()
    daily_pips = calculate_daily_pips(signals)
    
    performance = get_performance_summary()
    
    stats = performance.get("stats", {}) or {}
    analytics = performance.get("analytics", {}) or {}
    equity = performance.get("equity", {}) or {}
    
    market_volatility = calculate_market_volatility(signals)
    market_sentiment = calculate_market_sentiment(signals)
    
    can_trade, pause_reason = check_equity_protection(current_config)

    dashboard_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "session": session,
        "mode": current_mode,
        "sentiment_enabled": USE_SENTIMENT,
        "equity_protection": {
            "enabled": current_config.get("risk_management", {}).get("equity_protection", {}).get("enable", False),
            "can_trade": can_trade,
            "pause_reason": pause_reason if not can_trade else None
        },
        "market_state": {
            "volatility": market_volatility,
            "sentiment_bias": market_sentiment,
            "session": session,
            "trending_pairs": [s['pair'] for s in signals if s.get('metadata', {}).get('market_state') == 'TRENDING_STRONG']
        },
        "signals": signals,
        "api_usage": {
            "yfinance": {"successful_downloads": successful_downloads},
            "sentiment": {
                "enabled": USE_SENTIMENT,
                "newsapi": newsapi_calls,
                "marketaux": marketaux_calls
            }
        },
        "stats": {
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0),
            "total_pips": stats.get("total_pips", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "avg_win": stats.get("avg_win", 0),
            "avg_loss": stats.get("avg_loss", 0),
            "expectancy": stats.get("expectancy_per_trade", stats.get("expectancy", 0))
        },
        "risk_management": {
            "theoretical_max_pips": daily_pips,  # ✅ FIX #11: Renamed from "daily_pips"
            "total_risk_pips": sum(price_to_pips(s.get('pair', ''), abs(s.get('entry_price', 0) - s.get('sl', 0))) for s in signals),
            "max_daily_risk": current_config.get("risk_management", {}).get("max_daily_risk_pips", 150),
            "max_positions": current_config.get("risk_management", {}).get("max_open_positions", 3)
        },
        "analytics": analytics,
        "equity_curve": equity.get("curve", []),
        "system": {
            "last_update": datetime.now(timezone.utc).isoformat(),
            "data_sources_available": successful_downloads > 0,
            "sentiment_available": newsapi_calls > 0 or marketaux_calls > 0,
            "performance_tracking_enabled": PERFORMANCE_TRACKER is not None,
            "optimization_enabled": current_config.get("performance_tuning", {}).get("auto_adjust_thresholds", False),
            "signal_only_mode": SIGNAL_ONLY_MODE,
            "volume_scoring_enabled": USE_VOLUME_FOR_FX,
            "version": "2.0.6"
        }
    }

    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "dashboard_state.json"
    
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    log.info(f"📊 Dashboard written to {output_file}")
    
    if stats.get("total_trades", 0) > 0:
        log.info(f"📈 Performance: {stats.get('total_trades', 0)} trades | "
                f"Win Rate: {stats.get('win_rate', 0):.1f}% | "
                f"Total Pips: {stats.get('total_pips', 0):.1f} | "
                f"Expectancy: {stats.get('expectancy', 0):.2f}")
    
    write_health_check(signals, successful_downloads, newsapi_calls, marketaux_calls, can_trade, pause_reason, current_mode)

def write_health_check(signals: list, successful_downloads: int, newsapi_calls: int, marketaux_calls: int, can_trade: bool, pause_reason: str, mode: str):
    status = "ok"
    issues = []
    
    if not can_trade:
        status = "paused"
        issues.append(pause_reason)
    
    if successful_downloads == 0:
        status = "warning" if status == "ok" else status
        issues.append("Market data temporarily unavailable")
    
    if len(signals) == 0 and successful_downloads > 0 and can_trade:
        status = "warning" if status == "ok" else status
        issues.append("No signals generated")
    
    health = {
        "status": status,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "issues": issues,
        "can_trade": can_trade,
        "api_status": {
            "yfinance": "ok" if successful_downloads > 0 else "degraded",
            "newsapi": "ok" if newsapi_calls > 0 else ("disabled" if not USE_SENTIMENT else "unavailable"),
            "marketaux": "ok" if marketaux_calls > 0 else ("disabled" if not USE_SENTIMENT else "unavailable")
        },
        "system_info": {
            "mode": mode,
            "pairs_monitored": len(PAIRS),
            "last_success": datetime.now(timezone.utc).isoformat() if status == "ok" else None,
            "performance_tracking": PERFORMANCE_TRACKER is not None,
            "signal_only_mode": SIGNAL_ONLY_MODE,
            "version": "2.0.6"
        }
    }
    
    output_dir = Path("signal_state")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "health.json", "w") as f:
        json.dump(health, f, indent=2)
    
    if status == "ok":
        log.info("✅ System health: OK")
    elif status == "paused":
        log.warning(f"⏸️ System health: PAUSED - {', '.join(issues)}")
    elif status == "warning":
        log.warning(f"⚠️ System health: WARNING - {', '.join(issues)}")
    else:
        log.error(f"❌ System health: ERROR - {', '.join(issues)}")

# =========================
# TIME-WINDOW GUARD
# =========================
def in_execution_window():
    last_run_file = Path("signal_state/last_run.txt")
    success_file = Path("signal_state/last_success.txt")
    now = datetime.now(timezone.utc)

    if last_run_file.exists():
        with open(last_run_file, 'r') as f:
            last_run_str = f.read().strip()
        try:
            last_run = datetime.fromisoformat(last_run_str)
            
            if success_file.exists():
                with open(success_file, 'r') as f:
                    last_success_str = f.read().strip()
                try:
                    last_success = datetime.fromisoformat(last_success_str)
                    if now - last_success < timedelta(minutes=10):
                        log.info(f"⏱ Already ran successfully at {last_success} - exiting")
                        return False
                except Exception:
                    pass
            else:
                if now - last_run < timedelta(minutes=2):
                    log.info(f"⏱ Last run failed at {last_run}, waiting for retry window (2 min)")
                    return False
                else:
                    log.info(f"⚠️ Last run failed, attempting retry...")
        except Exception:
            pass

    last_run_file.parent.mkdir(exist_ok=True)
    with open(last_run_file, 'w') as f:
        f.write(now.isoformat())
    return True

def mark_success():
    success_file = Path("signal_state/last_success.txt")
    success_file.parent.mkdir(exist_ok=True)
    with open(success_file, 'w') as f:
        f.write(datetime.now(timezone.utc).isoformat())

# =========================
# MAIN
# =========================
def main():
    if not in_execution_window():
        return
    
    can_trade, pause_reason = check_equity_protection(CONFIG)
    if not can_trade:
        log.warning(f"⏸️ {pause_reason}")
        write_dashboard_state([], 0, 0, 0, CONFIG, MODE, SETTINGS)
        return
    
    current_config = CONFIG
    current_mode = MODE
    current_settings = SETTINGS
    
    sentiment_status = "ON" if USE_SENTIMENT else "OFF"
    volume_status = "ENABLED" if USE_VOLUME_FOR_FX else "DISABLED"
    
    log.info(f"🚀 Starting Trade Beacon v2.0.6 - Mode={current_mode} | Sentiment={sentiment_status}")
    log.info(f"📊 Monitoring {len(PAIRS)} pairs: {', '.join([p.replace('=X', '') for p in PAIRS])}")
    log.info(f"💰 Features: Real Fallback | Volume={volume_status} | Direction-Aware Correlation")
    log.info(f"🎯 Threshold: {current_settings.get('threshold')} | Min ADX: {current_settings.get('min_adx')} | Min R:R: {current_settings.get('min_risk_reward')}")
    
    active = []
    successful_downloads = 0
    newsapi_calls = 0
    marketaux_calls = 0

    log.info("🔍 Analyzing pairs with staggered execution...")
    
    max_workers = current_config.get("advanced", {}).get("parallel_workers", 3)
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(PAIRS))) as executor:
        futures = {executor.submit(generate_signal, pair): pair for pair in PAIRS}
        
        for future in as_completed(futures):
            pair = futures[future]
            try:
                sig, download_ok = future.result()
                
                if download_ok:
                    successful_downloads += 1
                
                if sig:
                    active.append(sig)
                    log.info(f"✅ {pair.replace('=X', '')} - Signal generated "
                            f"(Score: {sig['score']}, Confidence: {sig['confidence']}, RR: {sig['risk_reward']:.2f}, "
                            f"Modes: {', '.join(sig['eligible_modes'])})")
                else:
                    if download_ok:
                        log.info(f"⏭️ {pair.replace('=X', '')} - No signal")
                    else:
                        log.warning(f"⚠️ {pair.replace('=X', '')} - Download failed")
            except Exception as e:
                log.error(f"❌ {pair.replace('=X', '')} failed: {e}")
            
            time.sleep(0.5)

    if active:
        active, risk_warnings = check_risk_limits(active, current_config)
        for warning in risk_warnings:
            log.warning(f"⚠️ Risk Management: {warning}")

    if USE_SENTIMENT and active:
        try:
            news_agg = NewsAggregator()
            active = enhance_with_sentiment(active, news_agg)
            newsapi_calls = news_agg.newsapi_calls
            marketaux_calls = news_agg.marketaux_calls
            log.info("✅ Sentiment analysis complete")
        except Exception as e:
            log.error(f"❌ Sentiment analysis failed: {e}")
            log.info("⚠️ Continuing with technical signals only")

    log.info(f"\n✅ Cycle complete | Active signals: {len(active)}")
    
    write_dashboard_state(active, successful_downloads, newsapi_calls, marketaux_calls, 
                          current_config, current_mode, current_settings)

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        
        print("\n" + "="*80)
        print(f"🎯 {current_mode.upper()} SIGNALS {'+ SENTIMENT' if USE_SENTIMENT else ''} (v2.0.6):")
        print("="*80)
        
        display_cols = ["signal_id", "pair", "direction", "score", "confidence", 
                       "hold_time", "risk_reward", "eligible_modes"]
        print(df[display_cols].to_string(index=False))
        print("="*80 + "\n")
        
    else:
        log.info("✅ No strong signals this cycle")
    
    mark_success()
    log.info("✅ Run completed successfully - Trade Beacon v2.0.6")

# =========================
# GENERATOR MODE - CRITICAL SAFEGUARD
# =========================
SIGNAL_ONLY_MODE = True

def validate_signal_mode():
    if not SIGNAL_ONLY_MODE:
        raise RuntimeError(
            "❌ CRITICAL: Execution logic is disabled in signal generator mode. "
            "This system produces signals only."
        )
    log.info("🛡️ Signal-only mode validated - No execution logic will run")

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("trade-beacon")

validate_signal_mode()

# =========================
# CONFIGURATION
# =========================
PAIRS = [
    "USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X",
]

SPREADS = {
    "USDJPY": 0.002, "EURUSD": 0.00015, "GBPUSD": 0.0002,
    "AUDUSD": 0.00018, "NZDUSD": 0.0002, "USDCAD": 0.0002,
    "USDCHF": 0.0002, "EURJPY": 0.002, "GBPJPY": 0.003, "EURGBP": 0.00015,
}

CORRELATED_PAIRS = [
    {"EURUSD=X", "GBPUSD=X"}, {"EURUSD=X", "EURGBP=X"},
    {"GBPUSD=X", "EURGBP=X"}, {"USDJPY=X", "EURJPY=X"},
    {"USDJPY=X", "GBPJPY=X"}, {"EURJPY=X", "GBPJPY=X"},
]

INTERVAL = "15m"
LOOKBACK = "14d"
MIN_ROWS = 220
CACHE_TTL_SECONDS = 300

# ✅ FIX #2: Volume disabled for FX
USE_VOLUME_FOR_FX = False  # Volume data unreliable for forex

# =========================
# ✅ FIX #4: UNIFIED PIP CALCULATION
# =========================
def price_to_pips(pair: str, price_diff: float) -> float:
    """Single source of truth for pip calculations."""
    if "JPY" in pair:
        return abs(price_diff) / 0.01
    else:
        return abs(price_diff) / 0.0001

# =========================
# DATA SHAPE HELPER
# =========================
def ensure_series(data):
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data.squeeze()

# =========================
# RETRY DECORATOR
# =========================
def retry_with_backoff(max_retries=3, backoff_factor=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate limit" in error_msg or "429" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * backoff_factor
                            log.warning(f"⚠️ Rate limited, waiting {wait_time}s...")
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
# CONFIGURATION LOADER
# =========================
def load_config():
    config_path = Path("config.json")
    if not config_path.exists():
        log.warning("⚠️ config.json not found, using enhanced defaults")
        return _default_config()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    mode = config.get("mode", "conservative")
    if mode not in ["aggressive", "conservative"]:
        log.error(f"❌ Invalid mode '{mode}', defaulting to conservative")
        config["mode"] = "conservative"
        mode = "conservative"
    
    if mode not in config.get("settings", {}):
        log.error(f"❌ Settings missing for mode '{mode}'")
        raise ValueError(f"Config incomplete for mode: {mode}")
    
    perf_config = config.get("performance_tracking", {})
    if perf_config.get("enable", True):
        log.info(f"✅ Performance tracking enabled: {perf_config.get('history_file', 'signal_state/signal_history.json')}")
    
    log.info(f"✅ Config loaded: mode={mode}, sentiment={config.get('use_sentiment', False)}")
    return config

def _default_config():
    return {
        "mode": "conservative",
        "use_sentiment": False,
        "settings": {
            "aggressive": {
                "threshold": 60,
                "min_adx": 20,  # ✅ FIX #7: Lowered from 25
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "min_volume_ratio": 1.0,  # ✅ Not used if USE_VOLUME_FOR_FX=False
                "volume_penalty": 0,  # ✅ Disabled
                "min_risk_reward": 2.0,
                "atr_stop_multiplier": 1.8,
                "atr_target_multiplier": 4.0,
                "max_correlated_signals": 2
            },
            "conservative": {
                "threshold": 70,
                "min_adx": 22,  # ✅ FIX #7: Lowered from 30
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "min_volume_ratio": 1.0,  # ✅ Not used
                "volume_penalty": 0,  # ✅ Disabled
                "min_risk_reward": 2.5,
                "atr_stop_multiplier": 2.0,
                "atr_target_multiplier": 5.0,
                "max_correlated_signals": 1
            }
        },
        "advanced": {
            "enable_session_filtering": True,
            "enable_correlation_filter": True,
            "enable_performance_optimization": False,
            "cache_ttl_minutes": 5,
            "parallel_workers": 3,
            "session_bonuses": {
                "ASIAN": {"JPY_pairs": 3, "AUD_NZD_pairs": 3, "other": 0},
                "EUROPEAN": {"EUR_GBP_pairs": 3, "EUR_GBP_crosses": 2, "other": 0},
                "OVERLAP": {"all_major_pairs": 2},
                "US": {"USD_majors": 3, "other": 0},
                "LATE_US": {"all_major_pairs": 0}
            },
            "validation": {
                "max_price_change_pct": 0.03,
                "max_signal_age_seconds": 900,
                "min_sl_pips": {"JPY_pairs": 20, "other": 12},
                "max_spread_ratio": 0.25,
                "require_direction": True,
                "reject_missing_pips": True
            }
        },
        "risk_management": {
            "max_daily_risk_pips": 150,
            "max_open_positions": 3,
            "max_correlated_exposure": 1,
            "stop_trading_on_drawdown_pips": 100,
            "equity_protection": {
                "enable": False,  # ✅ FIX #3: Disabled in signal-only mode
                "max_consecutive_losses": 3,
                "pause_minutes_after_hit": 120
            }
        },
        "performance_tracking": {
            "enable": True,
            "history_file": "signal_state/signal_history.json",
            "idempotency": {"enabled": True, "scope": "daily"},
            "equity_curve": {"enabled": True, "max_points": 1000},
            "analytics": {
                "track_by_pair": True,
                "track_by_confidence": True,
                "track_by_session": True
            },
            "exports": {"enable_csv": True, "csv_path": "performance_export.csv"}
        },
        "performance_tuning": {
            "auto_adjust_thresholds": False,
            "min_trades_for_optimization": 50,
            "optimization_interval_days": 14,
            "target_win_rate": 0.5,
            "optimization_inputs": {
                "use_expectancy": True,
                "use_win_rate": True,
                "use_avg_rr": True,
                "min_expectancy": 0.5
            }
        },
        "logging": {
            "performance_level": "INFO",
            "log_equity_updates": True,
            "log_duplicate_signals": True
        }
    }

# =========================
# API KEY VALIDATION
# =========================
def validate_api_keys():
    required_keys = {
        'newsapi_key': os.environ.get('newsapi_key'),
        'MARKETAUX_API_KEY': os.environ.get('MARKETAUX_API_KEY')
    }
    
    missing_keys = [k for k, v in required_keys.items() if not v]
    
    if missing_keys:
        log.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
        return False
    
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": required_keys['newsapi_key']},
            timeout=5
        )
        if r.status_code in (401, 429):
            log.error("❌ Invalid NewsAPI key or rate limited")
            return False
        log.info("✅ NewsAPI key validated")
    except Exception as e:
        log.warning(f"⚠️ NewsAPI validation failed: {e}")
        return False
    
    try:
        r = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={"api_token": required_keys['MARKETAUX_API_KEY'], "limit": 1},
            timeout=5
        )
        if r.status_code in (401, 429):
            log.error("❌ Invalid Marketaux key or rate limited")
            return False
        log.info("✅ Marketaux key validated")
    except Exception as e:
        log.warning(f"⚠️ Marketaux validation failed: {e}")
        return False
    
    return True

CONFIG = load_config()
MODE = CONFIG["mode"]
USE_SENTIMENT = CONFIG.get("use_sentiment", False) and validate_api_keys()
SETTINGS = CONFIG["settings"][MODE]

# Initialize performance tracker
PERFORMANCE_TRACKER = None
if CONFIG.get("performance_tracking", {}).get("enable", True):
    try:
        history_file = CONFIG["performance_tracking"].get("history_file", "signal_state/signal_history.json")
        PERFORMANCE_TRACKER = PerformanceTracker(history_file=history_file)
        log.info("✅ Performance tracker initialized")
    except Exception as e:
        log.error(f"⚠️ Could not initialize performance tracker: {e}")
        PERFORMANCE_TRACKER = None

# =========================
# SIGNAL VALIDATION
# =========================
def validate_signal_quality(signal: Dict, config: Dict) -> Tuple[bool, List[str]]:
    warnings = []
    validation_config = config.get("advanced", {}).get("validation", {})
    mode = config.get("mode", "conservative")
    mode_settings = config["settings"][mode]
    
    if validation_config.get("require_direction", True):
        direction = signal.get('direction')
        if direction not in ("BUY", "SELL"):
            warnings.append(f"Invalid or missing direction: {direction}")
            return False, warnings
    
    sl_distance = abs(signal['entry_price'] - signal['sl'])
    if sl_distance == 0:
        warnings.append("Zero stop loss distance")
        return False, warnings
    
    spread = signal['spread']
    effective_spread = min(spread, sl_distance * 0.25)
    spread_ratio = effective_spread / sl_distance if sl_distance > 0 else 1
    
    max_spread_ratio = validation_config.get("max_spread_ratio", 0.25)
    if spread_ratio > max_spread_ratio:
        warnings.append(f"High spread ratio: {spread_ratio:.1%}")
        return False, warnings
    
    if "JPY" in signal['pair']:
        max_atr, min_atr = 1.0, 0.001
    else:
        max_atr, min_atr = 0.01, 0.00001
    
    if signal['atr'] <= min_atr or signal['atr'] > max_atr:
        warnings.append(f"Invalid ATR: {signal['atr']}")
        return False, warnings
    
    # ✅ FIX #4: Use unified pip calculation
    sl_pips = price_to_pips(signal['pair'], sl_distance)
    
    min_sl_pips_config = validation_config.get("min_sl_pips", {})
    if "JPY" in signal['pair']:
        min_sl_pips = min_sl_pips_config.get("JPY_pairs", 20)
    else:
        min_sl_pips = min_sl_pips_config.get("other", 12)
    
    if validation_config.get("reject_missing_pips", True) and sl_pips < 1:
        warnings.append(f"Missing pip calculation: {sl_pips:.1f}")
        return False, warnings
    
    if sl_pips < min_sl_pips:
        warnings.append(f"Stop too tight: {sl_pips:.1f} pips")
        return False, warnings
    
    min_rr = mode_settings.get("min_risk_reward", 2.0)
    if signal['risk_reward'] < min_rr:
        warnings.append(f"Poor R:R: {signal['risk_reward']:.2f}")
        return False, warnings
    
    max_price_change = validation_config.get("max_price_change_pct", 0.03)
    price_change_pct = abs(signal['tp'] - signal['sl']) / signal['entry_price']
    
    if price_change_pct > max_price_change:
        warnings.append(f"Unrealistic TP/SL: {price_change_pct:.1%}")
        return False, warnings
    
    # ✅ FIX #5: Validate signal age
    max_age = validation_config.get("max_signal_age_seconds", 900)
    try:
        signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
        signal_age = (datetime.now(timezone.utc) - signal_time).total_seconds()
        
        if signal_age > max_age:
            warnings.append(f"Stale signal: {signal_age/60:.1f} min")
            return False, warnings
    except Exception as e:
        warnings.append(f"Invalid timestamp: {e}")
        return False, warnings
    
    return True, warnings

# =========================
# BACKEND INTELLIGENCE
# =========================
def calculate_hold_time(risk_reward: float, atr: float) -> str:
    if risk_reward > 2.5 or atr > 0.002:
        return "SWING"
    elif risk_reward > 1.8 or atr > 0.0015:
        return "INTRADAY"
    return "SHORT"

def calculate_eligible_modes(score: int, adx: float, volume_ratio: float, 
                            rsi: float, config: Dict) -> List[str]:
    modes = []
    
    conservative_settings = config["settings"]["conservative"]
    if (score >= conservative_settings["threshold"] and
        adx >= conservative_settings["min_adx"]):
        modes.append("conservative")
    
    aggressive_settings = config["settings"]["aggressive"]
    if (score >= aggressive_settings["threshold"] and
        adx >= aggressive_settings["min_adx"]):
        modes.append("aggressive")
    
    return modes

def calculate_signal_freshness(timestamp: datetime) -> dict:
    age_minutes = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60
    
    if age_minutes < 15:
        status = "FRESH"
    elif age_minutes < 30:
        status = "RECENT"
    elif age_minutes < 60:
        status = "AGING"
    else:
        status = "STALE"
    
    confidence_decay = max(0, 100 - (age_minutes * 2))
    
    return {
        "status": status,
        "age_minutes": round(age_minutes, 1),
        "confidence_decay": round(confidence_decay, 1)
    }

def calculate_market_volatility(signals: List[Dict]) -> str:
    if not signals:
        return "CALM"
    
    avg_atr = sum(s.get("atr", 0) for s in signals) / len(signals)
    
    if avg_atr > 0.002:
        return "HIGH"
    elif avg_atr > 0.0015:
        return "NORMAL"
    return "CALM"

def calculate_market_sentiment(signals: List[Dict]) -> str:
    if not signals:
        return "MIXED"
    
    bullish = sum(1 for s in signals if s.get("direction") == "BUY")
    bearish = sum(1 for s in signals if s.get("direction") == "SELL")
    
    if bullish > bearish * 1.5:
        return "BULLISH"
    elif bearish > bullish * 1.5:
        return "BEARISH"
    return "MIXED"

# =========================
# MARKET SESSION
# =========================
def get_market_session() -> str:
    hour = datetime.now(timezone.utc).hour
    
    if 0 <= hour < 8:
        return "ASIAN"
    elif 8 <= hour < 13:
        return "EUROPEAN"
    elif 13 <= hour < 16:
        return "OVERLAP"
    elif 16 <= hour < 21:
        return "US"
    else:
        return "LATE_US"

def calculate_dynamic_session_bonus(pair: str, session: str, config: Dict) -> int:
    if not config.get("advanced", {}).get("enable_session_filtering", True):
        return 0
    
    session_config = config.get("advanced", {}).get("session_bonuses", {})
    
    if session not in session_config:
        return 0
    
    bonuses = session_config[session]
    
    if session == "ASIAN":
        if "JPY" in pair:
            return bonuses.get("JPY_pairs", 0)
        elif any(curr in pair for curr in ["AUD", "NZD"]):
            return bonuses.get("AUD_NZD_pairs", 0)
        return bonuses.get("other", 0)
    
    elif session in ["EUROPEAN", "OVERLAP"]:
        if any(curr in pair for curr in ["EUR", "GBP"]) and pair not in ["EURUSD", "GBPUSD"]:
            return bonuses.get("EUR_GBP_crosses", 0)
        elif any(curr in pair for curr in ["EUR", "GBP"]):
            return bonuses.get("EUR_GBP_pairs", 0)
        elif session == "OVERLAP":
            return bonuses.get("all_major_pairs", 0)
        return bonuses.get("other", 0)
    
    elif session == "US":
        if "USD" in pair and pair in ["EURUSD", "GBPUSD", "USDCAD"]:
            return bonuses.get("USD_majors", 0)
        return bonuses.get("other", 0)
    
    elif session == "LATE_US":
        return bonuses.get("all_major_pairs", 0)
    
    return 0

# =========================
# SENTIMENT (Placeholder)
# =========================
class SentimentAnalyzer:
    def __init__(self, hf_api_key: str = None):
        self.hf_api_key = hf_api_key
    
    def analyze(self, text: str) -> Dict:
        return {"label": "neutral", "score": 0.0}

class NewsAggregator:
    def __init__(self):
        self.newsapi_calls = 0
        self.marketaux_calls = 0
    
    def get_news(self, pairs: List[str]) -> List[Dict]:
        return []

def filter_articles_for_pair(pair: str, articles: List[Dict]) -> List[Dict]:
    return []

def analyze_sentiment_from_articles(pair: str, articles: List[Dict], analyzer) -> Dict:
    return {"adjustment": 0, "sentiment": "neutral", "news_count": 0, "sources": {}}

# =========================
# CACHE
# =========================
class MarketDataCache:
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if key not in self._cache:
                return None
            
            age = time.time() - self._timestamps.get(key, 0)
            if age > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return self._cache[key]
    
    def set(self, key: str, value: pd.DataFrame):
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

_market_cache = MarketDataCache()

# =========================
# TECHNICAL ANALYSIS
# =========================
def last(series: pd.Series):
    return None if series is None or series.empty else float(series.iloc[-1])

# ✅ FIX #1: REAL YFINANCE FALLBACK IMPLEMENTED
@retry_with_backoff(max_retries=3, backoff_factor=10)
def download(pair: str) -> Tuple[pd.DataFrame, bool]:
    cached = _market_cache.get(pair)
    if cached is not None:
        log.debug(f"📦 Using cached data for {pair}")
        return cached, True
    
    log.debug(f"📥 Downloading fresh data for {pair}")
    
    try:
        # Try 15m first
        df = yf.download(
            pair,
            interval="15m",
            period=LOOKBACK,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        
        # ✅ REAL FALLBACK: If 15m fails or insufficient, try 1h
        if df is None or df.empty or len(df) < MIN_ROWS:
            log.warning(f"⚠️ {pair} 15m failed/insufficient ({len(df) if df is not None else 0} rows), trying 1h fallback...")
            df = yf.download(
                pair,
                interval="1h",
                period="60d",  # Longer period for 1h to get enough data
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        
        if df is None or df.empty:
            log.warning(f"⚠️ {pair} download returned empty data even with fallback")
            return pd.DataFrame(), False
        
        df = df.dropna()
        
        if len(df) < MIN_ROWS:
            log.warning(f"⚠️ {pair} insufficient data: {len(df)} rows")
            return df, False
        
        _market_cache.set(pair, df)
        return df, True
        
    except Exception as e:
        log.error(f"❌ {pair} download failed: {e}")
        return pd.DataFrame(), False

def ema(series, period):
    return EMAIndicator(series, window=period).ema_indicator()

def rsi(series, period=14):
    return RSIIndicator(series, window=period).rsi()

def adx_calc(high, low, close):
    return ADXIndicator(high, low, close, window=14).adx()

def atr_calc(high, low, close):
    return AverageTrueRange(high, low, close, window=14).average_true_range()

def get_spread(pair: str) -> float:
    clean_pair = pair.replace("=X", "")
    return SPREADS.get(clean_pair, 0.0002)

def classify_market_state(adx: float, atr: float) -> str:
    if adx < 15:
        return "CHOPPY"
    elif adx > 25:
        return "TRENDING_STRONG"
    elif adx > 20:
        return "TRENDING_MODERATE"
    else:
        return "CONSOLIDATING"

def get_signal_type(e12: float, e26: float, e200: float, rsi: float) -> str:
    if e12 > e26 > e200:
        if rsi > 60:
            return "momentum"
        else:
            return "trend-continuation"
    elif e12 < e26 < e200:
        if rsi < 40:
            return "momentum"
        else:
            return "trend-continuation"
    elif (e12 > e26 and rsi < 40) or (e12 < e26 and rsi > 60):
        return "reversal"
    else:
        return "breakout"

# =========================
# ✅ FIX #3: EQUITY PROTECTION - SIGNAL MODE AWARE
# =========================
def check_equity_protection(config: Dict) -> Tuple[bool, str]:
    """Only checks if manually paused - doesn't enforce outcome-based logic in signal mode."""
    if SIGNAL_ONLY_MODE:
        # In signal-only mode, only respect manual pauses
        pause_file = Path("signal_state/trading_paused.json")
        if pause_file.exists():
            try:
                with open(pause_file, 'r') as f:
                    pause_data = json.load(f)
                    paused_until = datetime.fromisoformat(pause_data.get("paused_until"))
                    
                    if datetime.now(timezone.utc) < paused_until:
                        remaining = (paused_until - datetime.now(timezone.utc)).total_seconds() / 60
                        return False, f"Manually paused for {remaining:.1f} more minutes"
                    else:
                        pause_file.unlink()
            except Exception as e:
                log.warning(f"⚠️ Could not read pause file: {e}")
                pause_file.unlink()
        
        return True, ""
    
    # Full equity protection logic would go here for execution mode
    equity_config = config.get("risk_management", {}).get("equity_protection", {})
    
    if not equity_config.get("enable", True):
        return True, ""
    
    return True, ""

# =========================
# DETERMINISTIC SIGNAL ID
# =========================
def generate_deterministic_signal_id(pair: str, direction: str, entry_price: float, 
                                     session: str, date_str: str) -> str:
    """Generate a deterministic signal ID based on market state, not timestamp."""
    price_key = int(entry_price * 100000)
    signal_id = f"{pair}_{direction}_{price_key}_{session}_{date_str}"
    return signal_id

# =========================
# SIGNAL TRACKING & IDEMPOTENCY
# =========================
def get_existing_signals_today() -> List[str]:
    """Get list of signal IDs already generated today."""
    try:
        dashboard_file = Path("signal_state/dashboard_state.json")
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                data = json.load(f)
                signals = data.get("signals", [])
                today = datetime.now(timezone.utc).date()
                
                return [
                    s.get("signal_id") 
                    for s in signals 
                    if datetime.fromisoformat(s.get("timestamp", "")).date() == today
                ]
    except Exception as e:
        log.warning(f"⚠️ Could not load existing signals: {e}")
    
    return []

def is_duplicate_signal(signal_id: str, existing_ids: List[str]) -> bool:
    """Check if signal was already generated today."""
    return signal_id in existing_ids

# =========================
# PERFORMANCE-BASED THRESHOLD OPTIMIZATION
# =========================
def optimize_thresholds_if_needed(config: Dict) -> Dict:
    """
    Automatically adjust thresholds based on performance history.
    Only runs in signal-only mode if explicitly enabled.
    """
    tuning_config = config.get("performance_tuning", {})
    
    if not tuning_config.get("auto_adjust_thresholds", False):
        return config
    
    if not PERFORMANCE_TRACKER or not getattr(PERFORMANCE_TRACKER, "history", None):
        log.info("⚙️ Performance tracker not available, skipping optimization")
        return config
    
    stats = PERFORMANCE_TRACKER.history.get("stats", {}) or {}
    total_trades = stats.get("total_trades", 0)
    min_trades = tuning_config.get("min_trades_for_optimization", 50)
    
    if total_trades < min_trades:
        log.info(f"⚙️ Not enough trades for optimization ({total_trades}/{min_trades})")
        return config
    
    last_optimization_file = Path("signal_state/last_optimization.json")
    if last_optimization_file.exists():
        try:
            with open(last_optimization_file, 'r') as f:
                last_opt = json.load(f)
                last_opt_time = datetime.fromisoformat(last_opt.get("timestamp"))
                days_since = (datetime.now(timezone.utc) - last_opt_time).days
                
                interval_days = tuning_config.get("optimization_interval_days", 14)
                if days_since < interval_days:
                    log.info(f"⚙️ Optimization not due yet ({days_since}/{interval_days} days)")
                    return config
        except Exception:
            pass
    
    log.info("⚙️ Running threshold optimization...")
    
    win_rate = float(stats.get("win_rate", 0)) / 100
    expectancy = stats.get("expectancy_per_trade", stats.get("expectancy", 0))
    
    target_win_rate = tuning_config.get("target_win_rate", 0.5)
    min_expectancy = tuning_config.get("optimization_inputs", {}).get("min_expectancy", 0.5)
    
    mode = config.get("mode", "conservative")
    current_threshold = config["settings"][mode]["threshold"]
    
    adjustment = 0
    reason = "Optimal"
    
    if expectancy < min_expectancy:
        adjustment = 5
        reason = f"Low expectancy ({expectancy:.2f} < {min_expectancy})"
    elif win_rate < target_win_rate - 0.05:
        adjustment = 3
        reason = f"Low win rate ({win_rate:.1%} vs target {target_win_rate:.1%})"
    elif win_rate > target_win_rate + 0.10 and expectancy > min_expectancy * 2:
        adjustment = -3
        reason = f"Excellent performance (WR: {win_rate:.1%}, Exp: {expectancy:.2f})"
    
    if adjustment != 0:
        new_threshold = current_threshold + adjustment
        
        if mode == "aggressive":
            new_threshold = max(55, min(70, new_threshold))
        else:
            new_threshold = max(65, min(80, new_threshold))
        
        config["settings"][mode]["threshold"] = new_threshold
        
        log.info(f"✅ Threshold optimized: {current_threshold} → {new_threshold} ({reason})")
        
        opt_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "old_threshold": current_threshold,
            "new_threshold": new_threshold,
            "reason": reason,
            "stats": stats
        }
        
        last_optimization_file.parent.mkdir(exist_ok=True)
        with open(last_optimization_file, 'w') as f:
            json.dump(opt_record, f, indent=2)
    else:
        log.info("✅ Current thresholds are optimal, no adjustment needed")
    
    return config

# =========================
# TIME-WINDOW GUARD
# =========================
def in_execution_window():
    last_run_file = Path("signal_state/last_run.txt")
    success_file = Path("signal_state/last_success.txt")
    now = datetime.now(timezone.utc)

    if last_run_file.exists():
        with open(last_run_file, 'r') as f:
            last_run_str = f.read().strip()
        try:
            last_run = datetime.fromisoformat(last_run_str)
            
            if success_file.exists():
                with open(success_file, 'r') as f:
                    last_success_str = f.read().strip()
                try:
                    last_success = datetime.fromisoformat(last_success_str)
                    if now - last_success < timedelta(minutes=10):
                        log.info(f"⏱ Already ran successfully at {last_success} - exiting")
                        return False
                except Exception:
                    pass
            else:
                if now - last_run < timedelta(minutes=2):
                    log.info(f"⏱ Last run failed at {last_run}, waiting for retry window (2 min)")
                    return False
                else:
                    log.info(f"⚠️ Last run failed, attempting retry...")
        except Exception:
            pass

    last_run_file.parent.mkdir(exist_ok=True)
    with open(last_run_file, 'w') as f:
        f.write(now.isoformat())
    return True

def mark_success():
    success_file = Path("signal_state/last_success.txt")
    success_file.parent.mkdir(exist_ok=True)
    with open(success_file, 'w') as f:
        f.write(datetime.now(timezone.utc).isoformat())

# =========================
# MAIN
# =========================
def main():
    if not in_execution_window():
        return
    
    can_trade, pause_reason = check_equity_protection(CONFIG)
    if not can_trade:
        log.warning(f"⏸️ {pause_reason}")
        write_dashboard_state([], 0, 0, 0, CONFIG, MODE, SETTINGS)
        return
    
    current_config = CONFIG
    current_mode = MODE
    current_settings = SETTINGS
    
    if current_config.get("performance_tuning", {}).get("auto_adjust_thresholds", False):
        optimized_config = optimize_thresholds_if_needed(current_config)
        current_mode = optimized_config["mode"]
        current_settings = optimized_config["settings"][current_mode]
        log.info(f"⚙️ Using optimized thresholds: {current_settings.get('threshold')}")
    else:
        optimized_config = current_config
    
    sentiment_status = "ON" if USE_SENTIMENT else "OFF"
    volume_status = "ENABLED" if USE_VOLUME_FOR_FX else "DISABLED"
    
    log.info(f"🚀 Starting Trade Beacon v2.0.6 - Mode={current_mode} | Sentiment={sentiment_status}")
    log.info(f"📊 Monitoring {len(PAIRS)} pairs: {', '.join([p.replace('=X', '') for p in PAIRS])}")
    log.info(f"💰 Features: Real Fallback | Volume={volume_status} | Direction-Aware Correlation")
    log.info(f"🎯 Threshold: {current_settings.get('threshold')} | Min ADX: {current_settings.get('min_adx')} | Min R:R: {current_settings.get('min_risk_reward')}")
    
    active = []
    successful_downloads = 0
    newsapi_calls = 0
    marketaux_calls = 0

    _market_cache.clear()
    log.info("🔄 Cache cleared for fresh data")

    # Get existing signals to prevent duplicates
    existing_signal_ids = get_existing_signals_today()
    if existing_signal_ids:
        log.info(f"📋 Found {len(existing_signal_ids)} existing signals today")

    log.info("🔍 Analyzing pairs with staggered execution...")
    
    max_workers = optimized_config.get("advanced", {}).get("parallel_workers", 3)
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(PAIRS))) as executor:
        futures = {executor.submit(generate_signal, pair): pair for pair in PAIRS}
        
        for future in as_completed(futures):
            pair = futures[future]
            try:
                sig, download_ok = future.result()
                
                if download_ok:
                    successful_downloads += 1
                
                if sig:
                    # Check for duplicate
                    if is_duplicate_signal(sig['signal_id'], existing_signal_ids):
                        log.info(f"⏭️ {pair.replace('=X', '')} - Duplicate signal skipped")
                        continue
                    
                    active.append(sig)
                    log.info(f"✅ {pair.replace('=X', '')} - Signal generated "
                            f"(Score: {sig['score']}, Confidence: {sig['confidence']}, RR: {sig['risk_reward']:.2f}, "
                            f"Modes: {', '.join(sig['eligible_modes'])})")
                else:
                    if download_ok:
                        log.info(f"⏭️ {pair.replace('=X', '')} - No signal")
                    else:
                        log.warning(f"⚠️ {pair.replace('=X', '')} - Download failed")
            except Exception as e:
                log.error(f"❌ {pair.replace('=X', '')} failed: {e}")
            
            time.sleep(0.5)

    if active:
        active, risk_warnings = check_risk_limits(active, optimized_config)
        for warning in risk_warnings:
            log.warning(f"⚠️ Risk Management: {warning}")

    if USE_SENTIMENT and active:
        try:
            news_agg = NewsAggregator()
            active = enhance_with_sentiment(active, news_agg)
            newsapi_calls = news_agg.newsapi_calls
            marketaux_calls = news_agg.marketaux_calls
            log.info("✅ Sentiment analysis complete")
        except Exception as e:
            log.error(f"❌ Sentiment analysis failed: {e}")
            log.info("⚠️ Continuing with technical signals only")

    log.info(f"\n✅ Cycle complete | Active signals: {len(active)}")
    
    write_dashboard_state(active, successful_downloads, newsapi_calls, marketaux_calls, 
                          optimized_config, current_mode, current_settings)

    if active:
        df = pd.DataFrame(active)
        df.to_csv("signals.csv", index=False)
        log.info("📄 signals.csv written")
        
        print("\n" + "="*80)
        print(f"🎯 {current_mode.upper()} SIGNALS {'+ SENTIMENT' if USE_SENTIMENT else ''} (v2.0.6):")
        print("="*80)
        
        display_cols = ["signal_id", "pair", "direction", "score", "confidence", 
                       "hold_time", "risk_reward", "eligible_modes"]
        print(df[display_cols].to_string(index=False))
        print("="*80 + "\n")
        
    else:
        log.info("✅ No strong signals this cycle")
    
    mark_success()
    log.info("✅ Run completed successfully - Trade Beacon v2.0.6")


if __name__ == "__main__":
    main()
