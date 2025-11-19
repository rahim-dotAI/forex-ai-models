import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys
import os
import time
import re
from datetime import datetime

class ProgressTrackingExecutor(ExecutePreprocessor):
    """Custom executor with progress tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_cells = 0
        self.current_cell = 0
        self.start_time = None
        
    def preprocess(self, nb, resources=None, km=None):
        self.total_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
        self.current_cell = 0
        self.start_time = time.time()
        
        print(f"📊 Total code cells: {self.total_cells}")
        print("=" * 70)
        
        return super().preprocess(nb, resources, km)
    
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type != 'code':
            return cell, resources
        
        self.current_cell += 1
        elapsed = time.time() - self.start_time
        progress = (self.current_cell / self.total_cells) * 100
        
        print(f"\n🔄 Cell {self.current_cell}/{self.total_cells} ({progress:.1f}%) | ⏱️ {elapsed:.1f}s")
        
        cell_start = time.time()
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        cell_time = time.time() - cell_start
        
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    print(self._clean_output(output.text))
                elif output.output_type == 'error':
                    print(f"❌ {output.ename}: {output.evalue}")
        
        print(f"✅ Completed in {cell_time:.1f}s")
        
        return cell, resources
    
    def _clean_output(self, text):
        lines = [re.sub(r'\x1b\[[0-9;]*m', '', line) 
                for line in text.split('\n')
                if line.strip() and not any(x in line for x in ['[DEBUG]', 'WARNING:'])]
        return '\n'.join(lines)

def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ProgressTrackingExecutor(timeout=2400, kernel_name='python3', allow_errors=False)
    
    try:
        start = time.time()
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        duration = time.time() - start
        
        print(f"\n✅ Completed in {duration/60:.1f} min")
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    if not os.path.exists(notebook):
        print(f"❌ Not found: {notebook}")
        sys.exit(1)
    
    print("=" * 70)
    print("🧠 WEEKDAY FULL NOTEBOOK")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    
    success = run_notebook(notebook)
    sys.exit(0 if success else 1)
