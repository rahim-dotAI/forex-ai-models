import nbformat
import sys
import time
import json
import os
import re
from nbconvert.preprocessors import ExecutePreprocessor
from datetime import datetime, timezone

class WeekendContrarianExecutor(ExecutePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_count = 0
        self.start_time = None
        self.successful_cells = 0
        self.failed_cells = 0
        self.critical_errors = []
        self.stage_timings = {}
        self.current_stage = "Unknown"
        self.is_weekend = datetime.now(timezone.utc).weekday() in [5, 6]
    
    def preprocess(self, nb, resources=None, km=None):
        print("="*80)
        print("🚀 WEEKEND CONTRARIAN SCHEDULE v21.0 (1.5x Optimized)")
        print("="*80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        skip_av = os.environ.get('SKIP_ALPHA_VANTAGE', 'false').lower() == 'true'
        print(f"🔧 Alpha Vantage: {'SKIPPED ⏭️' if skip_av else 'ACTIVE ✅ (00:00 UTC)'}")
        
        if self.is_weekend:
            print("🏖️  WEEKEND MODE: Pipeline v6.3.1 Contrarian Active")
            print("   • Using 1.5x SL/TP (optimized for low volatility)")
            print("   • A/B Testing: 50% normal, 50% contrarian")
            print("   • Min eval window: 2-12 hours")
            print("   • Max timeout: 24-72 hours")
            print("   • Regime-aware Trade Beacon v21.0")
        else:
            print("💼 WEEKDAY MODE: Live trading with regime detection")
            print("   • Pipeline v6.3.1 normal mode")
            print("   • Using 2x normal SL/TP")
            print("   • Min eval window: 1-6 hours")
            print("   • Max timeout: 12-36 hours")
            print("   • Trade Beacon v21.0 with full regime detection")
        
        print(f"📊 Total cells to execute: {len([c for c in nb.cells if c.cell_type == 'code'])}")
        print(f"⏰ Trigger: Manual or colab_trigger.txt")
        print("="*80)
        print()
        
        self.start_time = time.time()
        return super().preprocess(nb, resources, km)
    
    def detect_stage(self, cell_source):
        source_lower = cell_source.lower()
        if 'api_keys' in source_lower or 'api keys' in source_lower:
            return "🔑 API Keys Setup"
        elif 'environment detection' in source_lower:
            return "🌍 Environment Detection"
        elif 'github sync' in source_lower:
            return "🔄 GitHub Sync"
        elif 'alpha vantage' in source_lower and 'fetcher' in source_lower:
            return "📈 Alpha Vantage Fetcher"
        elif 'yfinance' in source_lower and 'fetcher' in source_lower:
            return "📊 YFinance Fetcher"
        elif 'combiner' in source_lower:
            return "🔗 CSV Combiner"
        elif 'pipeline v6.3' in source_lower or 'weekend contrarian' in source_lower:
            return "🧠 Pipeline v6.3.1 Weekend Contrarian"
        elif 'trade beacon' in source_lower and 'v21' in source_lower:
            return "🌍 Trade Beacon v21.0 - Regime-Aware"
        elif 'learning' in source_lower and 'system' in source_lower:
            return "🎓 Adaptive Learning System"
        elif 'backtest' in source_lower:
            return "📉 Backtesting Module"
        return self.current_stage
    
    def preprocess_cell(self, cell, resources, idx):
        if cell.cell_type != 'code':
            return cell, resources
        
        self.cell_count += 1
        new_stage = self.detect_stage(cell.source)
        
        if new_stage != self.current_stage:
            if self.current_stage != "Unknown":
                duration = time.time() - self.stage_timings[self.current_stage]['start']
                print(f"   ⏱️  Stage completed in {duration:.1f}s")
                print()
            
            self.current_stage = new_stage
            self.stage_timings[new_stage] = {'start': time.time(), 'duration': 0}
            print("="*80)
            print(f"📍 STAGE: {new_stage}")
            if self.is_weekend and 'Pipeline' in new_stage:
                print("   🏖️  Weekend contrarian mode (1.5x SL/TP)")
            if 'Beacon' in new_stage:
                print("   🌍 Market regime detection active")
            print("="*80)
        
        elapsed = time.time() - self.start_time
        cell_start = time.time()
        
        preview = cell.source[:100].replace('\n', ' ')
        if len(cell.source) > 100:
            preview += "..."
        
        print(f"\n🔷 Cell {self.cell_count} | {int(elapsed)}s elapsed")
        print(f"   Code: {preview}")
        
        try:
            cell, resources = super().preprocess_cell(cell, resources, idx)
            cell_duration = time.time() - cell_start
            self.successful_cells += 1
            
            if cell.outputs:
                print(f"   ⏱️  Executed in {cell_duration:.2f}s")
                print(f"   📤 Output:")
                print("   " + "-"*70)
                
                output_lines = 0
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        text = re.sub(r'\x1b\[[0-9;]*m', '', output.text)
                        
                        for line in text.split('\n'):
                            if line.strip():
                                print(f"   │ {line}")
                                output_lines += 1
                                
                    elif output.output_type == 'execute_result':
                        if 'text/plain' in output.data:
                            result = output.data['text/plain']
                            print(f"   │ Result: {result}")
                            output_lines += 1
                            
                    elif output.output_type == 'error':
                        print(f"   │ ⚠️  Error: {output.ename}: {output.evalue}")
                        output_lines += 1
                
                print("   " + "-"*70)
                print(f"   ✅ Success ({output_lines} output lines)")
            else:
                print(f"   ✅ Success (no output) - {cell_duration:.2f}s")
            
        except Exception as e:
            cell_duration = time.time() - cell_start
            self.failed_cells += 1
            error_msg = str(e)
            print(f"   ❌ FAILED after {cell_duration:.2f}s")
            print(f"   │ Error: {error_msg[:200]}")
            
            if "CRITICAL" in error_msg.upper() or "FATAL" in error_msg.upper():
                self.critical_errors.append({
                    'cell': self.cell_count,
                    'stage': self.current_stage,
                    'error': error_msg[:200]
                })
        
        return cell, resources

# Load notebook
with open('AI_Forex_Brain_2.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print("\n" + "="*80)
print("🤖 FOREX AI BRAIN - WEEKEND CONTRARIAN v21.0")
print("="*80)
print(f"📓 Notebook: AI_Forex_Brain_2.ipynb")
print(f"🔧 Mode: Weekend Contrarian with Regime Detection")
print(f"⚙️  Pipeline: v6.3.1 (1.5x SL/TP optimization)")
print(f"⚙️  Trade Beacon: v21.0 (Market regime awareness)")
print(f"⏰ Trigger: Manual or colab_trigger.txt")
print("="*80)
print()

# Execute
ep = WeekendContrarianExecutor(timeout=2400, kernel_name='python3', allow_errors=True)
start = time.time()

try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    duration = time.time() - start
    
    print("\n" + "="*80)
    print("✅ EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"⏱️  Total Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"✅ Successful Cells: {ep.successful_cells}")
    print(f"❌ Failed Cells: {ep.failed_cells}")
    print(f"📊 Success Rate: {(ep.successful_cells/(ep.successful_cells+ep.failed_cells)*100):.1f}%")
    
    if ep.is_weekend:
        print(f"\n🏖️  Weekend Mode Summary:")
        print(f"   • Pipeline v6.3.1 contrarian active (1.5x SL/TP)")
        print(f"   • A/B testing 50% normal vs 50% contrarian")
        print(f"   • Faster evaluation with tighter stops")
        print(f"   • Trade Beacon v21.0 regime detection active")
    
    if ep.stage_timings:
        print("\n📊 Stage Timings:")
        for stage, timing in ep.stage_timings.items():
            duration = timing.get('duration', 0)
            if duration == 0:
                duration = time.time() - timing['start']
            print(f"   {stage}: {duration:.1f}s")
    
    if ep.critical_errors:
        print(f"\n⚠️  Critical Errors: {len(ep.critical_errors)}")
        for err in ep.critical_errors:
            print(f"   Cell {err['cell']} ({err['stage']}): {err['error'][:100]}")
    
    print("="*80 + "\n")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'trigger': 'manual_or_colab_trigger',
        'schedule_type': 'manual_trigger',
        'is_weekend': ep.is_weekend,
        'duration': duration,
        'cells_executed': ep.cell_count,
        'successful': ep.successful_cells,
        'failed': ep.failed_cells,
        'success_rate': round(ep.successful_cells/(ep.successful_cells+ep.failed_cells)*100, 2),
        'stage_timings': {k: v.get('duration', 0) for k, v in ep.stage_timings.items()},
        'critical_errors': len(ep.critical_errors),
        'status': 'success',
        'version': 'v21.0',
        'pipeline_version': 'v6.3.1',
        'beacon_version': 'v21.0'
    }
    
except Exception as e:
    duration = time.time() - start
    print("\n" + "="*80)
    print("❌ EXECUTION FAILED")
    print("="*80)
    print(f"Error: {str(e)[:300]}")
    print("="*80 + "\n")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'trigger': 'manual_or_colab_trigger',
        'schedule_type': 'manual_trigger',
        'is_weekend': ep.is_weekend,
        'duration': duration,
        'cells_executed': ep.cell_count,
        'status': 'error',
        'error': str(e)[:300],
        'version': 'v21.0'
    }

# Save report
os.makedirs('.github/run_history', exist_ok=True)
with open('.github/run_history/latest_run.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"💾 Run report saved to .github/run_history/latest_run.json")
