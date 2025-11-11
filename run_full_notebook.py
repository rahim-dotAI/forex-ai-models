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
        # Count code cells
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
        
        # Show first line of cell source (if not too long)
        if cell.source:
            first_line = cell.source.split('\n')[0][:60]
            if first_line.strip():
                print(f"📝 Cell preview: {first_line}...")
        
        print("-" * 70)
        
        # Execute the cell
        cell_start = time.time()
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        cell_duration = time.time() - cell_start
        
        # Show cell outputs (cleaned)
        if cell.outputs:
            for output in cell.outputs:
                # Handle different output types
                if output.output_type == 'stream':
                    text = output.text
                    # Clean up the output
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
                        # Show only last 5 lines of traceback
                        print("   Traceback (last 5 lines):")
                        for line in output.traceback[-5:]:
                            # Remove ANSI codes
                            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                            print(f"   {line}")
        
        # Show cell completion
        print("-" * 70)
        print(f"✅ Cell {self.current_cell}/{self.total_cells} completed in {cell_duration:.1f}s")
        print("=" * 70)
        
        return cell, resources
    
    def _clean_output(self, text):
        """Clean output text - remove debug logs, empty lines, etc."""
        if not text:
            return ""
        
        lines = []
        for line in text.split('\n'):
            # Skip debug/verbose lines
            if any(skip in line for skip in ['[DEBUG]', 'WARNING:', 'DeprecationWarning']):
                continue
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Remove ANSI color codes
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            lines.append(line)
        
        return '\n'.join(lines)

def run_notebook(notebook_path):
    """Execute entire Jupyter notebook with progress tracking"""
    print(f"📖 Reading notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    markdown_cells = sum(1 for c in nb.cells if c.cell_type == 'markdown')
    
    print(f"✅ Loaded notebook:")
    print(f"   • Total cells: {len(nb.cells)}")
    print(f"   • Code cells: {code_cells}")
    print(f"   • Markdown cells: {markdown_cells}")
    print()
    
    # Configure custom executor
    ep = ProgressTrackingExecutor(
        timeout=2400,  # 40 minutes per cell
        kernel_name='python3',
        allow_errors=False,
        store_widget_state=True
    )
    
    print("🚀 Starting notebook execution...")
    print()
    
    try:
        # Execute the notebook
        start_time = time.time()
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        total_duration = time.time() - start_time
        
        print()
        print("=" * 70)
        print("✅ NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⏱️  Total execution time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"📊 Cells executed: {code_cells}")
        print(f"⚡ Average time per cell: {total_duration/code_cells:.1f}s")
        print("=" * 70)
        
        # Save executed notebook
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"💾 Saved executed notebook: {output_path}")
        
        return True
        
    except CellExecutionError as e:
        print()
        print("=" * 70)
        print("❌ CELL EXECUTION FAILED!")
        print("=" * 70)
        
        # Extract cell information
        if hasattr(e, 'ename') and hasattr(e, 'evalue'):
            print(f"Error type: {e.ename}")
            print(f"Error message: {e.evalue}")
        
        if hasattr(e, 'traceback'):
            print("\nTraceback:")
            for line in e.traceback[-10:]:  # Last 10 lines
                line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                print(line)
        
        print("=" * 70)
        return False
    
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("=" * 70)
        return False

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    if not os.path.exists(notebook):
        print(f"❌ ERROR: Notebook not found: {notebook}")
        sys.exit(1)
    
    print("=" * 70)
    print("🧠 FOREX AI BRAIN - FULL NOTEBOOK EXECUTION")
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
