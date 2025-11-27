import sqlite3
import json
import os
from pathlib import Path

metrics = {'status': 'no_data', 'version': 'v20.2'}

# Pipeline database
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
            metrics['pipeline_win_rate'] = (result[1] / result[0] * 100) if result[0] else 0
        
        conn.close()
    except Exception as e:
        metrics['db_error'] = str(e)

# RL memory
rl_memory = Path('rl_memory/experience_replay.json.gz')
if rl_memory.exists():
    metrics['rl_memory_size'] = rl_memory.stat().st_size

# RL stats
stats_file = Path('rl_memory/learning_stats.json')
if stats_file.exists():
    try:
        with open(stats_file) as f:
            rl_stats = json.load(f)
        metrics['rl_trades'] = rl_stats.get('total_trades', 0)
        metrics['rl_win_rate'] = rl_stats.get('win_rate', 0) * 100
        metrics['rl_pnl'] = rl_stats.get('total_pnl', 0)
        metrics['epsilon'] = rl_stats.get('epsilon_history', [0.7])[-1] if rl_stats.get('epsilon_history') else 0.7
        
        # Multi-source stats
        metrics['pipeline_v6_learned'] = rl_stats.get('pipeline_v6_learned', 0)
        metrics['live_validated'] = rl_stats.get('live_validated', 0)
        metrics['backtest_learned'] = rl_stats.get('backtest_learned', 0)
        metrics['total_learning_sources'] = rl_stats.get('total_learning_sources', 0)
    except:
        pass

# Trade Beacon signals
beacon_file = Path('outputs/omega_signals.json')
if beacon_file.exists():
    try:
        with open(beacon_file) as f:
            beacon_data = json.load(f)
        metrics['beacon_iteration'] = beacon_data.get('iteration', 0)
        metrics['beacon_mode'] = beacon_data.get('mode', 'unknown')
        metrics['active_signals'] = sum(1 for s in beacon_data.get('signals', {}).values() if s.get('direction') != 'HOLD')
        
        # Learning summary
        if 'learning_summary' in beacon_data:
            metrics['learning_summary'] = beacon_data['learning_summary']
    except:
        pass

# Pipeline v6.1 learning outcomes
learning_db = Path('learning_data/learning_outcomes.json')
if learning_db.exists():
    try:
        with open(learning_db) as f:
            outcomes = json.load(f)
        metrics['learning_outcomes_count'] = len(outcomes)
        if outcomes:
            wins = sum(1 for o in outcomes if o.get('was_correct', False))
            metrics['learning_outcomes_wr'] = (wins / len(outcomes) * 100) if len(outcomes) else 0
    except:
        pass

# Pipeline v6.1 predictions
pred_file = Path('learning_data/predictions_history.json')
if pred_file.exists():
    try:
        with open(pred_file) as f:
            predictions = json.load(f)
        metrics['predictions_pending'] = sum(1 for p in predictions if not p.get('evaluated', True))
    except:
        pass

metrics['alpha_vantage_optimized'] = True
metrics['daily_api_calls'] = 4
metrics['api_usage_percent'] = 16
metrics['run_frequency'] = 'Every 2 hours'

os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics extracted:")
for key, value in metrics.items():
    print("  " + key + ": " + str(value))
