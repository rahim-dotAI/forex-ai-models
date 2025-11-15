#!/usr/bin/env python3
"""
Run complete Forex AI pipeline
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"

pipeline_steps = [
    ("1_fetch_alphavantage.py", "Fetching Alpha Vantage data"),
    ("2_fetch_yfinance.py", "Fetching YFinance data"),
    ("3_combine_csvs.py", "Combining and processing CSVs"),
    ("4_merge_pickles.py", "Merging pickle files"),
    ("5_train_pipeline.py", "Training models and generating signals"),
]

print("=" * 70)
print("🚀 FOREX AI PIPELINE - COMPLETE RUN")
print("=" * 70)

for i, (script, description) in enumerate(pipeline_steps, 1):
    print(f"\n[{i}/{len(pipeline_steps)}] {description}...")
    script_path = SCRIPTS_DIR / script
    
    if not script_path.exists():
        print(f"  ❌ Script not found: {script}")
        continue
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"  ✅ {script} completed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {script} failed with error code {e.returncode}")
        print(f"  Consider running manually: python scripts/{script}")
        sys.exit(1)

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📊 Check outputs/latest_signals.json for trading signals")
