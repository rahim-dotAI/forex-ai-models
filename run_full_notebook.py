import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
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
        
        # Show first line preview if it's meaningful
        if cell.source:
            first_line = cell.source.split('\n')[0][:60]
            if first_line.strip() and not first_line.strip().startswith('#'):
                print(f"📝 {first_line}...")
        
        cell_start = time.time()
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        cell_time = time.time() - cell_start
        
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    cleaned = self._clean_output(output.text)
                    if cleaned.strip():
                        print(cleaned)
                elif output.output_type == 'error':
                    print(f"❌ {output.ename}: {output.evalue}")
        
        print(f"✅ Completed in {cell_time:.1f}s")
        
        return cell, resources
    
    def _clean_output(self, text):
        if not text:
            return ""
        lines = []
        for line in text.split('\n'):
            # Skip debug/warning lines
            if any(skip in line for skip in ['[DEBUG]', 'WARNING:', 'DeprecationWarning']):
                continue
            if not line.strip():
                continue
            # Remove ANSI color codes
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
    
    print("🚀 Starting full notebook execution...")
    print()
    
    try:
        start = time.time()
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        duration = time.time() - start
        
        print()
        print("=" * 70)
        print("✅ FULL NOTEBOOK COMPLETED!")
        print("=" * 70)
        print(f"⏱️  Time: {duration:.1f}s ({duration/60:.1f} min)")
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
        print(f"❌ Not found: {notebook}")
        sys.exit(1)
    
    print("=" * 70)
    print("🧠 WEEKDAY FULL NOTEBOOK - TRADE BEACON v13.0")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"📓 {notebook}")
    print("🔴 Trade Beacon: Enhanced Production Edition")
    print("   • Market Regime Detection")
    print("   • Session-Based Trading")
    print("   • Momentum Confirmation")
    print("   • Strict Anti-Overfitting")
    print("=" * 70)
    print()
    
    success = run_notebook(notebook)
    
    print()
    print("=" * 70)
    if success:
        print("✅ WEEKDAY EXECUTION COMPLETED")
    else:
        print("❌ WEEKDAY EXECUTION FAILED")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
