import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone

metrics = {
    'status': 'no_data',
    'version': 'v21.1',
    'pipeline_version': 'v6.4.0',
    'beacon_version': 'v21.1',
    'trigger': 'manual_or_colab_trigger',
    'scheduler': 'manual_trigger',
    'is_weekend': datetime.now(timezone.utc).weekday() in [5, 6],
    'quality_filtering': 'active'
}

# Database metrics
db_path = Path('database/memory_v85.db')
if db_path.exists():
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("SELECT COUNT(*), SUM(CASE WHEN hit_tp THEN 1 ELSE 0 END) FROM completed_trades")
        result = c.fetchone()
        if result and result[0]:
            metrics['pipeline_trades'] = result[0]
            metrics['pipeline_wins'] = result[1] or 0
            metrics['pipeline_win_rate'] = round((result[1] or 0) / result[0] * 100, 2)
        conn.close()
        metrics['status'] = 'active'
    except Exception as e:
        metrics['db_error'] = str(e)[:100]

# RL Memory
rl_memory = Path('rl_memory/experience_replay.json.gz')
if rl_memory.exists():
    metrics['rl_memory_size'] = rl_memory.stat().st_size
    metrics['rl_memory_mb'] = round(rl_memory.stat().st_size / 1024 / 1024, 2)

# Learning stats
stats_file = Path('rl_memory/learning_stats.json')
if stats_file.exists():
    try:
        with open(stats_file) as f:
            rl_stats = json.load(f)
        metrics['rl_trades'] = rl_stats.get('total_trades', 0)
        metrics['rl_win_rate'] = round(rl_stats.get('win_rate', 0) * 100, 2)
        metrics['rl_pnl'] = round(rl_stats.get('total_pnl', 0.0), 2)
        metrics['epsilon'] = rl_stats.get('epsilon_history', [0.7])[-1] if rl_stats.get('epsilon_history') else 0.7
        metrics['regime_filtered'] = rl_stats.get('regime_filtered_trades', 0)
        metrics['quality_filtered'] = rl_stats.get('quality_filtered_trades', 0)
    except Exception as e:
        metrics['learning_stats_error'] = str(e)[:100]

# Quality weights
quality_weights_file = Path('quality_weights/learned_weights.json')
if quality_weights_file.exists():
    try:
        with open(quality_weights_file) as f:
            quality_data = json.load(f)
        metrics['quality_min_threshold'] = quality_data.get('min_quality_score', 80)
        metrics['quality_iterations'] = quality_data.get('learning_iterations', 0)
        metrics['quality_last_update'] = quality_data.get('last_updated', 'Never')
    except Exception as e:
        metrics['quality_error'] = str(e)[:100]

# Learning outcomes with weekend contrarian split
learning_db = Path('learning_data/learning_outcomes.json')
if learning_db.exists():
    try:
        with open(learning_db) as f:
            outcomes = json.load(f)
        
        metrics['total_outcomes'] = len(outcomes)
        
        # Split by weekend/weekday AND contrarian strategy
        weekend_normal = [o for o in outcomes if o.get('was_weekend_pred', False) and not o.get('is_contrarian', False)]
        weekend_contrarian = [o for o in outcomes if o.get('was_weekend_pred', False) and o.get('is_contrarian', False)]
        weekday_outcomes = [o for o in outcomes if not o.get('was_weekend_pred', False)]
        
        if weekend_normal:
            weekend_normal_wins = sum(1 for o in weekend_normal if o.get('was_correct', False))
            metrics['weekend_normal_outcomes'] = len(weekend_normal)
            metrics['weekend_normal_win_rate'] = round(weekend_normal_wins / len(weekend_normal) * 100, 2)
        
        if weekend_contrarian:
            weekend_contrarian_wins = sum(1 for o in weekend_contrarian if o.get('was_correct', False))
            metrics['weekend_contrarian_outcomes'] = len(weekend_contrarian)
            metrics['weekend_contrarian_win_rate'] = round(weekend_contrarian_wins / len(weekend_contrarian) * 100, 2)
        
        if weekday_outcomes:
            weekday_wins = sum(1 for o in weekday_outcomes if o.get('was_correct', False))
            metrics['weekday_outcomes'] = len(weekday_outcomes)
            metrics['weekday_win_rate'] = round(weekday_wins / len(weekday_outcomes) * 100, 2)
        
        # Check if adaptive windows are being used
        if outcomes and 'min_wait_hours' in outcomes[-1]:
            metrics['using_adaptive_windows'] = True
            last_outcome = outcomes[-1]
            metrics['last_min_wait'] = last_outcome.get('min_wait_hours', 0)
            metrics['last_max_wait'] = last_outcome.get('max_wait_hours', 0)
        else:
            metrics['using_adaptive_windows'] = False
        
    except Exception as e:
        metrics['outcomes_error'] = str(e)[:100]

# Predictions status
predictions_file = Path('learning_data/predictions_history.json')
if predictions_file.exists():
    try:
        with open(predictions_file) as f:
            predictions = json.load(f)
        
        pending = sum(1 for p in predictions if not p.get('evaluated', False))
        evaluated = sum(1 for p in predictions if p.get('evaluated', False))
        
        metrics['pending_predictions'] = pending
        metrics['evaluated_predictions'] = evaluated
        metrics['total_predictions'] = len(predictions)
        
        # Check oldest pending prediction age
        if predictions:
            pending_preds = [p for p in predictions if not p.get('evaluated', False)]
            if pending_preds:
                oldest = pending_preds[0]
                pred_time = datetime.fromisoformat(oldest['timestamp'].replace('Z', '+00:00'))
                hours_waiting = (datetime.now(timezone.utc) - pred_time).total_seconds() / 3600
                metrics['oldest_pending_hours'] = round(hours_waiting, 1)
        
    except Exception as e:
        metrics['predictions_error'] = str(e)[:100]

# Regime performance stats
regime_stats_file = Path('regime_stats/regime_performance.json')
if regime_stats_file.exists():
    try:
        with open(regime_stats_file) as f:
            regime_data = json.load(f)
        
        # Find best performing regimes
        best_regimes = []
        for key, data in regime_data.items():
            if data.get('total', 0) >= 5:
                win_rate = data['wins'] / data['total']
                best_regimes.append((key, win_rate, data['total']))
        
        best_regimes.sort(key=lambda x: x[1], reverse=True)
        metrics['best_regimes'] = [
            {'regime': r[0], 'win_rate': round(r[1] * 100, 2), 'trades': r[2]}
            for r in best_regimes[:5]
        ]
        
    except Exception as e:
        metrics['regime_error'] = str(e)[:100]

# Save metrics
os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print summary
print("\n" + "="*70)
print("📊 EXTRACTED METRICS (QUALITY FILTERED v21.1)")
print("="*70)
for key, value in metrics.items():
    print(f"{key}: {value}")
print("="*70)

# Quality filtering summary
if metrics.get('quality_filtered', 0) > 0:
    print("\n⭐ QUALITY FILTERING:")
    print("="*70)
    print(f"   Signals filtered: {metrics['quality_filtered']}")
    print(f"   Min threshold: {metrics.get('quality_min_threshold', 80)}")
    print(f"   Learning iterations: {metrics.get('quality_iterations', 0)}")
    print("="*70)

# Weekend-specific analysis
if metrics.get('is_weekend'):
    print("\n🏖️  WEEKEND CONTRARIAN A/B TEST:")
    print("="*70)
    
    if metrics.get('weekend_normal_outcomes', 0) > 0:
        print(f"   📊 NORMAL Strategy:")
        print(f"      Predictions: {metrics['weekend_normal_outcomes']}")
        print(f"      Win Rate: {metrics['weekend_normal_win_rate']}%")
    
    if metrics.get('weekend_contrarian_outcomes', 0) > 0:
        print(f"\n   🔄 CONTRARIAN Strategy:")
        print(f"      Predictions: {metrics['weekend_contrarian_outcomes']}")
        print(f"      Win Rate: {metrics['weekend_contrarian_win_rate']}%")
        
        # Compare
        if metrics.get('weekend_normal_outcomes', 0) > 0:
            normal_wr = metrics['weekend_normal_win_rate']
            contrarian_wr = metrics['weekend_contrarian_win_rate']
            diff = contrarian_wr - normal_wr
            
            print(f"\n   📈 Performance Comparison:")
            print(f"      Difference: {diff:+.2f}%")
            
            if diff > 20:
                print(f"      Status: 🎉 CONTRARIAN HUGE SUCCESS!")
            elif diff > 5:
                print(f"      Status: ✅ CONTRARIAN BETTER")
            elif diff < -5:
                print(f"      Status: ❌ CONTRARIAN WORSE")
            else:
                print(f"      Status: ⚖️  INCONCLUSIVE")
    
    if metrics.get('weekday_outcomes', 0) > 0:
        print(f"\n   💼 WEEKDAY Baseline:")
        print(f"      Predictions: {metrics['weekday_outcomes']}")
        print(f"      Win Rate: {metrics['weekday_win_rate']}%")
    
    print("="*70)

# Regime detection summary
if metrics.get('best_regimes'):
    print("\n🌍 TOP PERFORMING REGIMES:")
    print("="*70)
    for i, regime in enumerate(metrics['best_regimes'], 1):
        print(f"   {i}. {regime['regime']}")
        print(f"      Win Rate: {regime['win_rate']}%")
        print(f"      Trades: {regime['trades']}")
    print("="*70)

