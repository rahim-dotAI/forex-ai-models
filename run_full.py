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
        print("Processing cells...")
        print("Triggered by: Pipedream (exact 2-hour intervals)")
        
        skip_av = os.environ.get('SKIP_ALPHA_VANTAGE', 'false').lower() == 'true'
        print("Alpha Vantage:", "SKIPPED" if skip_av else "ACTIVE")
        
        self.start_time = time.time()
        return super().preprocess(nb, resources, km)
    
    def detect_stage(self, cell_source):
        source_lower = cell_source.lower()
        if 'api_keys' in source_lower:
            return "API Keys"
        elif 'environment detection' in source_lower:
            return "Environment"
        elif 'github sync' in source_lower:
            return "GitHub Sync"
        elif 'alpha vantage' in source_lower and 'fetcher' in source_lower:
            return "Alpha Vantage"
        elif 'yfinance' in source_lower and 'fetcher' in source_lower:
            return "YFinance"
        elif 'combiner' in source_lower:
            return "CSV Combiner"
        elif 'pipeline v6' in source_lower:
            return "Pipeline v6.1"
        elif 'trade beacon' in source_lower:
            return "Trade Beacon v20.2"
        return self.current_stage
    
    def preprocess_cell(self, cell, resources, idx):
        if cell.cell_type != 'code':
            return cell, resources
        
        self.cell_count += 1
        new_stage = self.detect_stage(cell.source)
        
        if new_stage != self.current_stage:
            if self.current_stage in self.stage_timings:
                self.stage_timings[self.current_stage]['duration'] = time.time() - self.stage_timings[self.current_stage]['start']
            self.current_stage = new_stage
            self.stage_timings[new_stage] = {'start': time.time(), 'duration': 0}
            print(f"STAGE: {new_stage}")
        
        elapsed = time.time() - self.start_time
        print(f"Cell {self.cell_count} | {self.current_stage} | {int(elapsed)}s")
        
        try:
            cell, resources = super().preprocess_cell(cell, resources, idx)
            self.successful_cells += 1
            
            if cell.outputs:
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        text = re.sub(r'\x1b\[[0-9;]*m', '', output.text)
                        important = ['COMPLETE', 'ERROR', 'Win Rate', 'LEARNING', 
                                   'Evaluated', 'learning_outcomes', 'Pipeline v6']
                        for line in text.split('\n'):
                            if any(m in line for m in important):
                                print(f"  {line}")
        except Exception as e:
            self.failed_cells += 1
            print(f"  Error: {str(e)[:100]}")
        
        return cell, resources

with open('AI_Forex_Brain_2.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print("="*70)
print("PIPEDREAM-TRIGGERED LEARNING SYSTEM")
print("="*70)

ep = SmartExecutor(timeout=2400, kernel_name='python3', allow_errors=True)
start = time.time()

try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    duration = time.time() - start
    
    print(f"\n{'='*70}")
    print("EXECUTION COMPLETED")
    print(f"Duration: {round(duration, 1)}s")
    print(f"Success: {ep.successful_cells}, Failed: {ep.failed_cells}")
    print(f"{'='*70}\n")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'trigger': 'pipedream',
        'duration': duration,
        'cells_executed': ep.cell_count,
        'successful': ep.successful_cells,
        'failed': ep.failed_cells,
        'status': 'success'
    }
except Exception as e:
    duration = time.time() - start
    print(f"\nError: {str(e)[:200]}")
    report = {
        'timestamp': datetime.now().isoformat(),
        'trigger': 'pipedream',
        'duration': duration,
        'status': 'error',
        'error': str(e)[:200]
    }

os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/latest_run.json', 'w') as f:
    json.dump(report, f, indent=2)
