import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys, os, time, re, json
from datetime import datetime

class SummaryExecutor(ExecutePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_count = 0
        self.start_time = None
        self.key_outputs = []
        
    def preprocess(self, nb, resources=None, km=None):
        print(f"📊 Processing {len(nb.cells)} cells...")
        self.start_time = time.time()
        return super().preprocess(nb, resources, km)
    
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type != 'code':
            return cell, resources
        
        self.cell_count += 1
        
        if self.cell_count % 5 == 0:
            elapsed = time.time() - self.start_time
            print(f"⏳ Progress: {self.cell_count} cells ({elapsed:.0f}s elapsed)")
        
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        
        # Collect key outputs
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    text = re.sub(r'\x1b\[[0-9;]*m', '', output.text)
                    lines = text.strip().split('\n')
                    for line in lines:
                        if any(marker in line for marker in ['✅', '⚠️', '❌', '💰', '🧠', 'COMPLETE', 'Iteration', 'Win Rate', 'Total P&L']):
                            print(f"   {line}")
                            self.key_outputs.append(line)
        
        return cell, resources

def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = SummaryExecutor(timeout=2400, kernel_name='python3', allow_errors=False)
    
    print("="*70)
    print("🚀 STARTING FULL NOTEBOOK EXECUTION")
    print("="*70)
    
    start = time.time()
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    duration = time.time() - start
    
    print("\n" + "="*70)
    print(f"✅ COMPLETED: {ep.cell_count} cells in {duration:.1f}s")
    print("="*70)
    
    # Save run report
    report = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'full_notebook',
        'total_cells': ep.cell_count,
        'duration': duration,
        'key_outputs': ep.key_outputs[-20:]
    }
    
    os.makedirs('.github/run_history', exist_ok=True)
    with open('.github/run_history/latest_run.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return True

if __name__ == "__main__":
    notebook = "AI_Forex_Brain_2.ipynb"
    if not os.path.exists(notebook):
        print(f"❌ Notebook not found: {notebook}")
        sys.exit(1)
    
    try:
        success = run_notebook(notebook)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        sys.exit(1)
