import sqlite3
import json
import os
from pathlib import Path

metrics = {'status': 'no_data', 'version': 'v20.2', 'trigger': 'pipedream'}

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
        conn.close()
    except:
        pass

rl_memory = Path('rl_memory/experience_replay.json.gz')
if rl_memory.exists():
    metrics['rl_memory_size'] = rl_memory.stat().st_size

stats_file = Path('rl_memory/learning_stats.json')
if stats_file.exists():
    try:
        with open(stats_file) as f:
            rl_stats = json.load(f)
        metrics['rl_trades'] = rl_stats.get('total_trades', 0)
        metrics['rl_win_rate'] = rl_stats.get('win_rate', 0) * 100
        metrics['pipeline_v6_learned'] = rl_stats.get('pipeline_v6_learned', 0)
        metrics['live_validated'] = rl_stats.get('live_validated', 0)
    except:
        pass

learning_db = Path('learning_data/learning_outcomes.json')
if learning_db.exists():
    try:
        with open(learning_db) as f:
            outcomes = json.load(f)
        metrics['learning_outcomes_count'] = len(outcomes)
    except:
        pass

os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
