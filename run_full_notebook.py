import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys
import os
import time
import re
from datetime import datetime

class ProgressTrackingExecutor(ExecutePreprocessor):
    """Custom executor that shows progress and only cell outputs"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_cells = 0
        self.current_cell = 0
        self.start_time = None
        
    def preprocess(self, nb, resources=None, km=None):
        """Override to track total cells"""
        self.total_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
        self.current_cell = 0
        self.start_time = time.time()
        
        print(f"📊 Total code cells to execute: {self.total_cells}")
        print("=" * 70)
        print()
        
        return super().preprocess(nb, resources, km)
    
    def preprocess_cell(self, cell, resources, cell_index):
        """Override to show progress before executing each cell"""
        if cell.cell_type != 'code':
            return cell, resources
        
        self.current_cell += 1
        elapsed = time.time() - self.start_time
        
        # Calculate progress
        progress_pct = (self.current_cell / self.total_cells) * 100
        
        # Show cell header with progress
        print()
        print("=" * 70)
        print(f"🔄 EXECUTING CELL {self.current_cell}/{self.total_cells} " 
              f"({progress_pct:.1f}% complete)")
        print(f"⏱️  Elapsed time: {elapsed:.1f}s")
        
        # Show first line of cell source
        if cell.source:
            first_line = cell.source.split('\n')[0][:60]
            if first_line.strip() and not first_line.strip().startswith('#'):
                print(f"📝 Cell preview: {first_line}...")
        
        print("-" * 70)
        
        # Execute the cell
        cell_start = time.time()
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        cell_duration = time.time() - cell_start
        
        # Show cell outputs (cleaned)
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    text = output.text
                    text = self._clean_output(text)
                    if text.strip():
                        print(text)
                
                elif output.output_type == 'execute_result':
                    if 'text/plain' in output.data:
                        text = output.data['text/plain']
                        text = self._clean_output(text)
                        if text.strip():
                            print(text)
                
                elif output.output_type == 'error':
                    print(f"❌ ERROR in cell {self.current_cell}:")
                    print(f"   {output.ename}: {output.evalue}")
                    if output.traceback:
                        for line in output.traceback[-5:]:
                            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                            print(f"   {line}")
        
        # Show cell completion
        print("-" * 70)
        print(f"✅ Cell {self.current_cell}/{self.total_cells} completed in {cell_duration:.1f}s")
        print("=" * 70)
        
        return cell, resources
    
    def _clean_output(self, text):
        """Clean output text"""
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
    """Execute entire Jupyter notebook with progress tracking"""
    print(f"📖 Reading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    
    print(f"✅ Loaded notebook:")
    print(f"   • Total cells: {len(nb.cells)}")
    print(f"   • Code cells: {code_cells}")
    print()
    
    ep = ProgressTrackingExecutor(
        timeout=2400,  # 40 minutes per cell
        kernel_name='python3',
        allow_errors=False,
        store_widget_state=True
    )
    
    print("🚀 Starting full notebook execution...")
    print()
    
    try:
        start_time = time.time()
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        total_duration = time.time() - start_time
        
        print()
        print("=" * 70)
        print("✅ FULL NOTEBOOK EXECUTION COMPLETED!")
        print("=" * 70)
        print(f"⏱️  Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"📊 Cells executed: {code_cells}")
        print(f"⚡ Avg time/cell: {total_duration/code_cells:.1f}s")
        print("=" * 70)
        
        return True
        
    except CellExecutionError as e:
        print()
        print("=" * 70)
        print("❌ CELL EXECUTION FAILED!")
        print("=" * 70)
        print(f"Error: {e.ename}: {e.evalue}")
        print("=" * 70)
        return False
    
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 70)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print("=" * 70)
        return False

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    if not os.path.exists(notebook):
        print(f"❌ ERROR: Notebook not found: {notebook}")
        sys.exit(1)
    
    print("=" * 70)
    print("🧠 WEEKDAY MODE - FULL NOTEBOOK EXECUTION")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"📓 Notebook: {notebook}")
    print("=" * 70)
    print()
    
    success = run_notebook(notebook)
    
    print()
    print("=" * 70)
    if success:
        print("✅ EXECUTION COMPLETED SUCCESSFULLY")
    else:
        print("❌ EXECUTION FAILED")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
