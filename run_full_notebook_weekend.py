import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys
import os
import time
import re
from datetime import datetime

class ProgressTrackingExecutor(ExecutePreprocessor):
    """Custom executor that shows progress"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_cells = 0
        self.current_cell = 0
        self.start_time = None
        
    def preprocess(self, nb, resources=None, km=None):
        self.total_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
        self.current_cell = 0
        self.start_time = time.time()
        
        print(f"📊 Total code cells to execute: {self.total_cells}")
        print("=" * 70)
        print()
        
        return super().preprocess(nb, resources, km)
    
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type != 'code':
            return cell, resources
        
        self.current_cell += 1
        elapsed = time.time() - self.start_time
        progress_pct = (self.current_cell / self.total_cells) * 100
        
        print()
        print("=" * 70)
        print(f"🔄 CELL {self.current_cell}/{self.total_cells} ({progress_pct:.1f}%)")
        print(f"⏱️  Elapsed: {elapsed:.1f}s")
        
        if cell.source:
            first_line = cell.source.split('\n')[0][:60]
            if first_line.strip() and not first_line.strip().startswith('#'):
                print(f"📝 {first_line}...")
        
        print("-" * 70)
        
        cell_start = time.time()
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        cell_duration = time.time() - cell_start
        
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    text = self._clean_output(output.text)
                    if text.strip():
                        print(text)
                elif output.output_type == 'error':
                    print(f"❌ ERROR: {output.ename}: {output.evalue}")
        
        print("-" * 70)
        print(f"✅ Cell {self.current_cell} completed in {cell_duration:.1f}s")
        print("=" * 70)
        
        return cell, resources
    
    def _clean_output(self, text):
        if not text:
            return ""
        lines = []
        for line in text.split('\n'):
            if any(skip in line for skip in ['[DEBUG]', 'WARNING:', 'DeprecationWarning']):
                continue
            if not line.strip():
                continue
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            lines.append(line)
        return '\n'.join(lines)

def run_notebook(notebook_path):
    print(f"📖 Reading: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f"✅ Loaded: {len(nb.cells)} cells ({code_cells} code)")
    print()
    
    ep = ProgressTrackingExecutor(
        timeout=2400,
        kernel_name='python3',
        allow_errors=False
    )
    
    print("🚀 Starting full notebook execution (data generation)...")
    print()
    
    try:
        start_time = time.time()
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        total_duration = time.time() - start_time
        
        print()
        print("=" * 70)
        print("✅ FULL NOTEBOOK COMPLETED!")
        print("=" * 70)
        print(f"⏱️  Time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"📊 Cells: {code_cells}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ EXECUTION FAILED!")
        print("=" * 70)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print("=" * 70)
        return False

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    if not os.path.exists(notebook):
        print(f"❌ Notebook not found: {notebook}")
        sys.exit(1)
    
    print("=" * 70)
    print("📓 WEEKEND DATA GENERATION MODE")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"📓 {notebook}")
    print("🎯 Purpose: Create data files for future learning runs")
    print("=" * 70)
    print()
    
    success = run_notebook(notebook)
    
    print()
    print("=" * 70)
    if success:
        print("✅ DATA GENERATION COMPLETED")
        print("💡 Next weekend: Will use tagged cells only!")
    else:
        print("❌ DATA GENERATION FAILED")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
