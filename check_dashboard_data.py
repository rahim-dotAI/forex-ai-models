#!/usr/bin/env python3
"""
Dashboard Data Diagnostic Tool
Checks what data is in dashboard_state.json and signal_history.json
"""

import json
from pathlib import Path

print("=" * 60)
print("TRADE BEACON DASHBOARD DIAGNOSTIC")
print("=" * 60)

# Check dashboard_state.json
dashboard_file = Path("signal_state/dashboard_state.json")
if dashboard_file.exists():
    print("\n✅ dashboard_state.json EXISTS")
    with open(dashboard_file) as f:
        data = json.load(f)
    
    print(f"\n📊 Keys in dashboard_state.json:")
    for key in data.keys():
        print(f"   - {key}")
    
    print(f"\n📈 Stats:")
    print(f"   - Total Trades: {data.get('stats', {}).get('total_trades', 0)}")
    print(f"   - Win Rate: {data.get('stats', {}).get('win_rate', 0)}%")
    print(f"   - Total Pips: {data.get('stats', {}).get('total_pips', 0)}")
    
    print(f"\n📡 Active Signals:")
    print(f"   - Aggressive: {data.get('active_signals_by_mode', {}).get('aggressive', 0)}")
    print(f"   - Conservative: {data.get('active_signals_by_mode', {}).get('conservative', 0)}")
    
    print(f"\n📜 Historical Signals:")
    hist_signals = data.get('historical_signals', [])
    print(f"   - Count: {len(hist_signals)}")
    if hist_signals:
        print(f"   - Sample:")
        for sig in hist_signals[:3]:
            print(f"     • {sig.get('pair')} {sig.get('direction')} - {sig.get('status')} ({sig.get('pips_result', 0):.1f} pips)")
    
    print(f"\n🔍 Performance Stats:")
    perf_stats = data.get('performance_stats', {})
    if perf_stats:
        print(f"   - Total Trades: {perf_stats.get('total_trades', 0)}")
        print(f"   - Wins: {perf_stats.get('wins', 0)}")
        print(f"   - Losses: {perf_stats.get('losses', 0)}")
        print(f"   - Win Rate: {perf_stats.get('win_rate', 0):.1f}%")
        
        print(f"\n   📊 By Mode:")
        by_mode = perf_stats.get('by_mode', {})
        for mode, data in by_mode.items():
            print(f"     • {mode}: {data.get('trades', 0)} trades, {data.get('win_rate', 0):.1f}% WR")
        
        print(f"\n   🏆 By Tier:")
        by_tier = perf_stats.get('by_tier', {})
        for tier, data in by_tier.items():
            print(f"     • {tier}: {data.get('trades', 0)} trades, {data.get('win_rate', 0):.1f}% WR")
    else:
        print("   ❌ No performance_stats found!")
    
else:
    print("\n❌ dashboard_state.json DOES NOT EXIST")

# Check signal_history.json
print("\n" + "=" * 60)
history_file = Path("signal_state/signal_history.json")
if history_file.exists():
    print("✅ signal_history.json EXISTS")
    with open(history_file) as f:
        history = json.load(f)
    
    signals = history.get('signals', [])
    stats = history.get('stats', {})
    
    print(f"\n📊 Total Signals: {len(signals)}")
    print(f"\n📈 Stats:")
    print(f"   - Total Trades: {stats.get('total_trades', 0)}")
    print(f"   - Wins: {stats.get('wins', 0)}")
    print(f"   - Losses: {stats.get('losses', 0)}")
    print(f"   - Win Rate: {stats.get('win_rate', 0):.1f}%")
    
    if signals:
        print(f"\n📜 Signals Breakdown:")
        status_counts = {}
        for sig in signals:
            status = sig.get('status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            print(f"   - {status}: {count}")
        
        print(f"\n🎯 Sample Signals:")
        for sig in signals[:3]:
            print(f"   • {sig.get('pair')} {sig.get('direction')} - {sig.get('status')} - {sig.get('timestamp', 'No timestamp')}")
else:
    print("❌ signal_history.json DOES NOT EXIST")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
