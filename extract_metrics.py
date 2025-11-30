import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone

metrics = {
    'status': 'no_data', 
    'version': 'v20.4-adaptive',
    'trigger': 'manual_or_colab_trigger',
    'scheduler': 'manual_trigger',
    'is_weekend': datetime.now(timezone.utc).weekday() in [5, 6]
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
        metrics['rl_pnl'] = round(rl_stats.get('total_pnl', 0), 2)
    except Exception as e:
        metrics['learning_stats_error'] = str(e)[:100]

# Learning outcomes with weekend split
learning_db = Path('learning_data/learning_outcomes.json')
if learning_db.exists():
    try:
        with open(learning_db) as f:
            outcomes = json.load(f)
        
        metrics['total_outcomes'] = len(outcomes)
        
        # Split by weekend/weekday
        weekend_outcomes = [o for o in outcomes if o.get('was_weekend_pred', False)]
        weekday_outcomes = [o for o in outcomes if not o.get('was_weekend_pred', False)]
        
        if weekend_outcomes:
            weekend_wins = sum(1 for o in weekend_outcomes if o.get('was_correct', False))
            metrics['weekend_outcomes'] = len(weekend_outcomes)
            metrics['weekend_win_rate'] = round(weekend_wins / len(weekend_outcomes) * 100, 2)
        
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

# Save metrics
os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print summary
print("\n" + "="*70)
print("📊 EXTRACTED METRICS (ADAPTIVE WEEKEND MODE)")
print("="*70)
for key, value in metrics.items():
    print(f"{key}: {value}")
print("="*70)

# Weekend-specific analysis
if metrics.get('is_weekend'):
    print("\n🏖️  WEEKEND ANALYSIS:")
    print("="*70)
    if metrics.get('weekend_outcomes', 0) > 0:
        print(f"   Weekend Predictions: {metrics['weekend_outcomes']}")
        print(f"   Weekend Win Rate: {metrics['weekend_win_rate']}%")
    if metrics.get('weekday_outcomes', 0) > 0:
        print(f"   Weekday Predictions: {metrics['weekday_outcomes']}")
        print(f"   Weekday Win Rate: {metrics['weekday_win_rate']}%")
    
    if metrics.get('using_adaptive_windows'):
        print(f"\n   ✅ Adaptive Windows: ACTIVE")
        print(f"   Last min wait: {metrics.get('last_min_wait', 0)}h")
        print(f"   Last max wait: {metrics.get('last_max_wait', 0)}h")
    else:
        print(f"\n   ⚠️  Adaptive Windows: NOT DETECTED")
        print(f"   Consider updating to Pipeline v6.2 Pro")
    
    if metrics.get('pending_predictions', 0) > 0:
        print(f"\n   📊 Pending: {metrics['pending_predictions']} predictions")
        print(f"   Oldest waiting: {metrics.get('oldest_pending_hours', 0)}h")
        
        oldest_hours = metrics.get('oldest_pending_hours', 0)
        if oldest_hours >= 2:
            print(f"   ✅ Ready for evaluation (>2h wait)")
        else:
            print(f"   ⏳ Needs more time ({2 - oldest_hours:.1f}h remaining)")
    
    print("="*70)
