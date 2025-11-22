import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys, os, time, re, traceback
from datetime import datetime

class TaggedCellExecutor(ExecutePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tagged_cells = []
        self.current_cell = 0
        self.start_time = None
        self.cell_summaries = []
        
    def preprocess(self, nb, resources=None, km=None):
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and '#TAG: pipeline_main' in cell.source:
                self.tagged_cells.append(idx)
        
        print(f"📊 Found {len(self.tagged_cells)} tagged cells")
        self.start_time = time.time()
        return super().preprocess(nb, resources, km)
    
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type != 'code' or cell_index not in self.tagged_cells:
            return cell, resources
        
        self.current_cell += 1
        print(f"\n{'='*70}")
        print(f"🔄 TAGGED CELL {self.current_cell}/{len(self.tagged_cells)}")
        print(f"{'='*70}")
        
        cell_start = time.time()
        
        try:
            cell, resources = super().preprocess_cell(cell, resources, cell_index)
            cell_time = time.time() - cell_start
            
            # Extract key output lines (not the full code)
            output_lines = []
            if cell.outputs:
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        text = re.sub(r'\x1b\[[0-9;]*m', '', output.text)
                        lines = text.strip().split('\n')
                        # Only show important lines
                        for line in lines:
                            if any(marker in line for marker in ['✅', '⚠️', '❌', '💰', '🧠', '💾', '📊', 'Iteration', 'Mode:', 'COMPLETE', 'Win Rate', 'Total P&L']):
                                output_lines.append(line)
                    elif output.output_type == 'error':
                        # Extract only the error name and message, NOT the traceback
                        error_name = output.get('ename', 'Error')
                        error_value = output.get('evalue', 'Unknown error')
                        output_lines.append(f"❌ {error_name}: {error_value}")
            
            # Print condensed output
            if output_lines:
                print("\n📋 Key Output:")
                for line in output_lines[:30]:  # Limit to 30 most important lines
                    print(f"   {line}")
                if len(output_lines) > 30:
                    print(f"   ... ({len(output_lines) - 30} more lines)")
            
            print(f"\n✅ Cell {self.current_cell} completed in {cell_time:.1f}s")
            
            self.cell_summaries.append({
                'cell': self.current_cell,
                'duration': cell_time,
                'status': 'success',
                'key_outputs': len(output_lines)
            })
            
        except Exception as e:
            cell_time = time.time() - cell_start
            
            # Extract ONLY the error type and message, no code
            error_name = type(e).__name__
            error_msg = str(e)
            
            # If it's a CellExecutionError, extract the real error
            if 'CellExecutionError' in error_name:
                # Try to extract just the error name and value
                if hasattr(e, 'ename') and hasattr(e, 'evalue'):
                    error_name = e.ename
                    error_msg = e.evalue
                else:
                    # Parse from string if needed
                    error_lines = str(e).split('\n')
                    for line in error_lines:
                        if ': ' in line and not line.strip().startswith('#'):
                            parts = line.split(': ', 1)
                            if len(parts) == 2:
                                error_name = parts[0].strip()
                                error_msg = parts[1].strip()
                                break
            
            print(f"\n❌ Cell {self.current_cell} FAILED after {cell_time:.1f}s")
            print(f"   Error Type: {error_name}")
            print(f"   Error Message: {error_msg[:200]}")  # Limit error message length
            
            self.cell_summaries.append({
                'cell': self.current_cell,
                'duration': cell_time,
                'status': 'failed',
                'error': f"{error_name}: {error_msg[:100]}"
            })
            
            # Re-raise but we've already printed the clean version
            raise RuntimeError(f"Cell {self.current_cell} execution failed: {error_name}")
        
        return cell, resources

def run_tagged_cells(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = TaggedCellExecutor(timeout=2400, kernel_name='python3', allow_errors=False)
    
    print("="*70)
    print("🚀 STARTING TAGGED CELLS EXECUTION")
    print("="*70)
    
    start = time.time()
    
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
    except Exception as e:
        # Error already printed in preprocess_cell, just pass through
        pass
    
    duration = time.time() - start
    
    # Print summary
    print("\n" + "="*70)
    print("📊 EXECUTION SUMMARY")
    print("="*70)
    
    failed_cells = []
    for summary in ep.cell_summaries:
        status_icon = "✅" if summary['status'] == 'success' else "❌"
        if summary['status'] == 'failed':
            failed_cells.append(summary['cell'])
            print(f"{status_icon} Cell {summary['cell']}: {summary['duration']:.1f}s - FAILED")
            print(f"   └─ {summary.get('error', 'Unknown error')}")
        else:
            print(f"{status_icon} Cell {summary['cell']}: {summary['duration']:.1f}s - {summary.get('key_outputs', 0)} key outputs")
    
    print(f"\n⏱️  Total Duration: {duration:.1f}s")
    print("="*70)
    
    if failed_cells:
        print(f"\n❌ Execution completed with errors in cells: {', '.join(map(str, failed_cells))}")
        return False
    else:
        print(f"\n✅ All cells executed successfully")
        return True

if __name__ == "__main__":
    notebook = "AI_Forex_Brain_2.ipynb"
    if not os.path.exists(notebook):
        print(f"❌ Notebook not found: {notebook}")
        sys.exit(1)
    
    try:
        success = run_tagged_cells(notebook)
        sys.exit(0 if success else 1)
    except Exception as e:
        # Clean error message only
        print(f"\n❌ FATAL ERROR: {type(e).__name__}")
        sys.exit(1)
