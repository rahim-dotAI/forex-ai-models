import sqlite3
import json
import os
from pathlib import Path

metrics = {
    'status': 'no_data', 
    'version': 'v20.3-pipedream',
    'trigger': 'github_actions',
    'scheduler': 'pipedream_schedule',
    'hours': 'even (0,2,4,6,8,10,12,14,16,18,20,22)'
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
        metrics['pipeline_v6_learned'] = rl_stats.get('pipeline_v6_learned', 0)
        metrics['live_validated'] = rl_stats.get('live_validated', 0)
    except Exception as e:
        metrics['learning_stats_error'] = str(e)[:100]

# Learning outcomes
learning_db = Path('learning_data/learning_outcomes.json')
if learning_db.exists():
    try:
        with open(learning_db) as f:
            outcomes = json.load(f)
        metrics['learning_outcomes_count'] = len(outcomes)
        
        # Recent outcomes analysis
        if outcomes:
            recent = outcomes[-10:] if len(outcomes) >= 10 else outcomes
            wins = sum(1 for o in recent if o.get('outcome') == 'win')
            metrics['recent_win_rate'] = round(wins / len(recent) * 100, 2)
    except Exception as e:
        metrics['outcomes_error'] = str(e)[:100]

# Save metrics
os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print summary
print("\n" + "="*70)
print("📊 EXTRACTED METRICS")
print("="*70)
for key, value in metrics.items():
    print(f"{key}: {value}")
print("="*70)
