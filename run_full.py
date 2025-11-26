import nbformat
import sys
import time
import json
import os
import re
from nbconvert.preprocessors import ExecutePreprocessor
from datetime import datetime

class SmartExecutor(ExecutePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_count = 0
        self.start_time = None
        self.successful_cells = 0
        self.failed_cells = 0
        self.critical_errors = []
        self.stage_timings = {}
        self.current_stage = "Unknown"
    
    def preprocess(self, nb, resources=None, km=None):
        code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
        print("Processing cells...")
        print("Environment: GitHub Actions")
        
        skip_av = os.environ.get('SKIP_ALPHA_VANTAGE', 'false').lower() == 'true'
        if skip_av:
            print("Alpha Vantage: SKIPPED")
        else:
            print("Alpha Vantage: ACTIVE")
        
        self.start_time = time.time()
        return super().preprocess(nb, resources, km)
    
    def detect_stage(self, cell_source):
        source_lower = cell_source.lower()
        if 'api_keys' in source_lower or 'alpha_vantage_key' in source_lower:
            return "API Keys"
        elif 'environment detection' in source_lower or 'in_colab' in source_lower:
            return "Environment"
        elif 'github sync' in source_lower or 'repo_folder' in source_lower:
            return "GitHub Sync"
        elif 'alpha vantage' in source_lower and 'fetcher' in source_lower:
            return "Alpha Vantage"
        elif 'yfinance' in source_lower and 'fetcher' in source_lower:
            return "YFinance"
        elif 'combiner' in source_lower or 'indicator' in source_lower:
            return "CSV Combiner"
        elif 'ultra-persistent' in source_lower or 'pipeline' in source_lower:
            return "ML Pipeline"
        elif 'trade beacon' in source_lower or 'deep q-learning' in source_lower:
            return "RL Agent"
        return self.current_stage
    
    def preprocess_cell(self, cell, resources, idx):
        if cell.cell_type != 'code':
            return cell, resources
        
        self.cell_count += 1
        
        new_stage = self.detect_stage(cell.source)
        if new_stage != self.current_stage:
            stage_start = time.time()
            if self.current_stage in self.stage_timings:
                self.stage_timings[self.current_stage]['duration'] = stage_start - self.stage_timings[self.current_stage]['start']
            self.current_stage = new_stage
            self.stage_timings[new_stage] = {'start': stage_start, 'duration': 0}
            print("STAGE: " + new_stage)
        
        elapsed = time.time() - self.start_time
        print("Cell " + str(self.cell_count) + " | " + self.current_stage + " | " + str(int(elapsed)) + "s elapsed")
        
        try:
            cell, resources = super().preprocess_cell(cell, resources, idx)
            self.successful_cells += 1
            
            if cell.outputs:
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        text = re.sub(r'\x1b\[[0-9;]*m', '', output.text)
                        lines = text.strip().split('\n')
                        
                        important_markers = [
                            'COMPLETE', 'ERROR', 'FAILED', 'SUCCESS', 'SKIPPED',
                            'Win Rate', 'P&L', 'Iteration', 'Mode:', 'Total', 'Average',
                            'Loaded', 'Saved', 'Updated', 'Found', 'Processing',
                            'Quality', 'Trades', 'Epsilon', 'Experience Replay',
                            'Pipeline Stats', 'Database', 'Q-Network', 'Backtest',
                            'API calls', 'Daily API usage', 'Alpha Vantage'
                        ]
                        
                        important_lines = [l for l in lines if any(marker in l for marker in important_markers)]
                        
                        if important_lines:
                            for line in important_lines[:15]:
                                print("  " + line)
                    
                    elif output.output_type == 'error':
                        error_msg = output.ename + ": " + output.evalue
                        print("  Error: " + error_msg)
                        self.critical_errors.append({
                            'cell': self.cell_count,
                            'stage': self.current_stage,
                            'error': error_msg
                        })
        
        except Exception as e:
            self.failed_cells += 1
            error_summary = str(e)[:200]
            print("  Cell " + str(self.cell_count) + " error: " + error_summary)
            print("  Continuing to next cell...")
            self.critical_errors.append({
                'cell': self.cell_count,
                'stage': self.current_stage,
                'error': error_summary
            })
        
        return cell, resources

if not os.path.exists('AI_Forex_Brain_2.ipynb'):
    print("ERROR: AI_Forex_Brain_2.ipynb not found!")
    sys.exit(1)

with open('AI_Forex_Brain_2.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print("=" * 70)
print("FOREX AI BRAIN - FULL PIPELINE EXECUTION")
print("=" * 70)

ep = SmartExecutor(timeout=2400, kernel_name='python3', allow_errors=True)
start = time.time()

try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    duration = time.time() - start
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED")
    print("=" * 70)
    print("Duration: " + str(round(duration, 1)) + "s (" + str(round(duration/60, 1)) + " min)")
    print("Cells: " + str(ep.cell_count) + " total")
    print("Success: " + str(ep.successful_cells))
    print("Failed: " + str(ep.failed_cells))
    
    if ep.stage_timings:
        print("\nStage Timings:")
        for stage, timing in ep.stage_timings.items():
            duration_val = timing.get('duration', 0)
            if duration_val > 0:
                print("  " + stage + ": " + str(round(duration_val, 1)) + "s")
    
    if ep.critical_errors:
        print("\nCritical Errors (" + str(len(ep.critical_errors)) + "):")
        for err in ep.critical_errors[:5]:
            print("  Cell " + str(err['cell']) + " (" + err['stage'] + "): " + err['error'][:100])
    
    print("=" * 70 + "\n")
    
    report = {
        'timestamp': datetime.now().isoformat(), 
        'mode': 'full_pipeline', 
        'duration': duration,
        'cells_executed': ep.cell_count,
        'successful': ep.successful_cells,
        'failed': ep.failed_cells,
        'status': 'success',
        'stage_timings': {k: v['duration'] for k, v in ep.stage_timings.items() if 'duration' in v},
        'critical_errors': len(ep.critical_errors),
        'alpha_vantage_active': os.environ.get('SKIP_ALPHA_VANTAGE', 'false').lower() != 'true'
    }
    
except Exception as e:
    duration = time.time() - start
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED WITH ERRORS")
    print("=" * 70)
    print("Error: " + type(e).__name__ + ": " + str(e)[:200])
    print("Duration: " + str(round(duration, 1)) + "s (" + str(round(duration/60, 1)) + " min)")
    print("Successful cells: " + str(ep.successful_cells))
    print("Failed cells: " + str(ep.failed_cells))
    print("=" * 70 + "\n")
    
    report = {
        'timestamp': datetime.now().isoformat(), 
        'mode': 'full_pipeline', 
        'duration': duration,
        'cells_executed': ep.cell_count,
        'successful': ep.successful_cells,
        'failed': ep.failed_cells,
        'status': 'completed_with_errors',
        'error': type(e).__name__ + ": " + str(e)[:200],
        'stage_timings': {k: v['duration'] for k, v in ep.stage_timings.items() if 'duration' in v},
        'critical_errors': len(ep.critical_errors),
        'alpha_vantage_active': os.environ.get('SKIP_ALPHA_VANTAGE', 'false').lower() != 'true'
    }

os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/latest_run.json', 'w') as f: 
    json.dump(report, f, indent=2)

print("Full notebook execution completed")
print("Report saved to .github/run_history/latest_run.json")
